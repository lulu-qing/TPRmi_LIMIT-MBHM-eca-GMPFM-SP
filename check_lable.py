import sqlite3
import os

# 根据 mbhm.py 的默认路径，数据库通常在这个位置
db_path = '/data1/zhangyong/lq/mbhm_dataset/metadata.sqlite'
if not os.path.exists(db_path):
    print(f"找不到数据库文件: {db_path}")
    exit()

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 使用 JOIN 语句联合查询 condition 表和 file_info 表
query = """
SELECT c.dataset, f.label 
FROM file_info f
JOIN condition c ON f.condition_id = c.condition_id
GROUP BY c.dataset, f.label
"""

cursor.execute(query)
results = cursor.fetchall()

dataset_groups = {}
for dataset_name, label in results:
    if dataset_name not in dataset_groups:
        dataset_groups[dataset_name] = []
    dataset_groups[dataset_name].append(label)

print("====== 请将以下字典直接复制替换到 fscil_trainer.py 中 ======")
print("dataset_groups = {")
for name, labels in dataset_groups.items():
    labels.sort()
    print(f"    '{name}': {labels},")
print("}")
print("============================================================")

conn.close()