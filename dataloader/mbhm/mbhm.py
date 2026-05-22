import os
import sys
import torch
import numpy as np
import h5py
import sqlite3
from torch.utils.data import Dataset

# 获取当前文件的绝对路径，向上退两级到达根目录，并加入到系统路径中
# 这行代码解决了 "No module named 'functions'" 的问题
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from functions.dcn import dcn

class MBHM(Dataset):
    def __init__(self, root, train=True, index=None, index_path=None, base_sess=None):
        self.root = root
        self.train = train

        # 1. 确定需要加载的【原始类别列表】
        if index_path is not None:
            with open(index_path, 'r') as f:
                self.class_indices = [int(line.strip()) for line in f.readlines()]
        elif index is not None:
            self.class_indices = list(index)
        else:
            self.class_indices = None

        # ==============================================================
        # [核心修复]：建立从 "原始跳跃标签" 到 "连续标签(0-9)" 的映射表
        # 读取所有的 session txt 文件，按顺序分配 0,1,2...9
        self.label_map = {}
        map_idx = 0
        for i in range(1, 5):  # 你的增量阶段共 4 个 session (1,2,3,4)
            txt_p = f"data/index_list/mbhm/session_{i}.txt"
            if os.path.exists(txt_p):
                with open(txt_p, 'r') as f:
                    for line in f.readlines():
                        orig_label = int(line.strip())
                        if orig_label not in self.label_map:
                            self.label_map[orig_label] = map_idx
                            map_idx += 1
        # ==============================================================

        # 2. 读取 SQLite 数据库获取标签
        meta_path = os.path.join(self.root, 'metadata.sqlite')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"找不到数据库文件: {meta_path}")

        conn = sqlite3.connect(meta_path)
        cursor = conn.cursor()

        # 修改：使用联合查询，同时带出文件对应的具体数据集名称 (c.dataset)
        cursor.execute('''
            SELECT f.file_id, f.label, c.dataset
            FROM file_info f
            JOIN condition c ON f.condition_id = c.condition_id
        ''') 
        all_meta = cursor.fetchall()
        conn.close()

        # 3. 过滤类别并划分 Train/Test
        subset_meta = []
        # 修改：正确解包 3 个变量 (包含 dataset_name)
        for idx, (file_id, label, dataset_name) in enumerate(all_meta):
            if self.class_indices is not None and label not in self.class_indices:
                continue

            is_train = (idx % 10 < 9)

            if self.train and is_train:
                subset_meta.append((file_id, label, dataset_name)) 
            elif not self.train and not is_train:
                subset_meta.append((file_id, label, dataset_name)) 

        # 4. 读取 HDF5 获取真实的振动信号
        self.data = []
        self.targets = []
        self.file_ids = []         # 用于存储纯数字的文件标识符
        self.dataset_names = []    # 用于存储对应的具体数据集名称 (如 'CWRU', 'PU')
        hdf5_path = os.path.join(self.root, 'data.hdf5')

        with h5py.File(hdf5_path, 'r') as f:
            vib_data = f['vibration']
            for file_id, label, dataset_name in subset_meta: 
                self.data.append(vib_data[file_id])
                self.targets.append(self.label_map[label])
                self.dataset_names.append(dataset_name)
                self.file_ids.append(file_id)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        signal = self.data[i]
        target = self.targets[i]

        signal = np.array(signal)
        signal = dcn(signal)

        signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)

        return signal_tensor, target

# ================= 测试代码 =================
if __name__ == '__main__':
    # 这里的 root 路径请根据你实际的 data 存放位置调整
    dataset = MBHM(root='/data1/zhangyong/lq/mbhm_dataset/', train=False) 
    
    print(f"成功加载数据集，总样本数: {len(dataset)}")
    
    # 打印前 5 个样本的信息，检查名字是否被正确读取
    for i in range(5):
        signal, target = dataset[i]
        file_id = dataset.file_ids[i] 
        d_name = dataset.dataset_names[i]
        print(f"样本 {i}: 映射后标签={target}, 真实数据集={d_name}, file_id={file_id}, 信号形状={signal.shape}")