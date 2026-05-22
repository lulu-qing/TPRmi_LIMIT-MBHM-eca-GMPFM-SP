import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base.Network import MYNET as Net
import numpy as np
from copy import deepcopy
import math

import math

class SPAMC(nn.Module):
    def __init__(self, embed_dim):
        super(SPAMC, self).__init__()
        self.embed_dim = embed_dim
        # Semantic Attention 的线性投影层 (Eq. 20)
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # 学习尺度因子 η (Eq. 22)
        self.eta = nn.Parameter(torch.tensor(0.1)) 
        
    def forward(self, C_m, lambda_prior):
        """
        C_m: 上下文集合 [Batch, N_classes + 1, Embed_dim] (包含 W_m-1, P_new, phi_q)
        lambda_prior: 尺度先验集合 [Batch, N_classes + 1, Gate_dim]
        """
        # 1. 语义流 (Semantic Stream) - Eq. 20
        Q = self.W_q(C_m)
        K = self.W_k(C_m)
        V = self.W_v(C_m)
        A_sem = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.embed_dim)
        
        # 2. 先验流 (Scale Prior Stream) - Eq. 21
        # 计算尺度一致性矩阵 (余弦相似度)
        lambda_norm = F.normalize(lambda_prior, p=2, dim=-1)
        A_prior = torch.bmm(lambda_norm, lambda_norm.transpose(1, 2))
        
        # 3. 协同校准 (Calibration) - Eq. 22
        A_calib = F.softmax(A_sem + self.eta * A_prior, dim=-1)
        
        # 4. 加权聚合 - Eq. 23
        C_calib = torch.bmm(A_calib, V)
        
        return C_calib

class WeightECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(WeightECA, self).__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, w):
        y = self.avg_pool(w)
        y = y.transpose(-1, -2)
        y = self.conv(y)
        y = self.sigmoid(y.transpose(-1, -2))
        return w * y


class ECA_RefConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, groups=1, map_k=3):
        super(ECA_RefConv1d, self).__init__()
        assert map_k <= kernel_size

        self.origin_kernel_shape = (out_channels, in_channels // groups, kernel_size)
        self.register_buffer('weight', torch.zeros(*self.origin_kernel_shape))

        self.num_kernels = out_channels * in_channels // groups
        G = in_channels * out_channels // (groups ** 2)

        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = padding if padding is not None else kernel_size // 2
        self.bias = None

        self.convmap1 = nn.Conv1d(
            in_channels=self.num_kernels, out_channels=self.num_kernels,
            kernel_size=map_k, stride=1, padding=map_k // 2, groups=G, bias=False
        )
        self.eca = WeightECA(channels=self.num_kernels)
        self.convmap2 = nn.Conv1d(
            in_channels=self.num_kernels, out_channels=self.num_kernels,
            kernel_size=map_k, stride=1, padding=map_k // 2, groups=G, bias=False
        )

        nn.init.dirac_(self.convmap1.weight)
        # 赋予微小扰动，打破零梯度僵局
        #nn.init.normal_(self.convmap2.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.convmap2.weight)

    def get_equivalent_kernel(self):
        # 强制将权重转到输入数据所在的设备上
        # self.weight 是 buffer，它是随模型在 GPU 上的
        origin_weight = self.weight.view(1, self.num_kernels, self.kernel_size)
        
        # 确保 convmap1/2 的参数也同步到正确设备
        # 在 forward 之前加这一步保护：
        if self.convmap1.weight.device != self.weight.device:
            self.convmap1.to(self.weight.device)
            self.convmap2.to(self.weight.device)
            self.eca.to(self.weight.device)

        w1 = self.convmap1(origin_weight)
        w2 = self.eca(w1)
        w_vi = self.weight + self.convmap2(w2).view(*self.origin_kernel_shape)
        return w_vi

    def forward(self, inputs):
        kernel = self.get_equivalent_kernel()
        return F.conv1d(inputs, kernel, stride=self.stride, padding=self.padding, groups=self.groups, bias=self.bias)


def sample_task_ids(support_label, num_task, num_shot, num_way, num_class):
    basis_matrix = torch.arange(num_shot).long().view(-1, 1).repeat(1, num_way).view(-1) * num_class
    permuted_ids = torch.zeros(num_task, num_shot * num_way).long()
    permuted_labels = []
    for i in range(num_task):
        clsmap = torch.randperm(num_class)[:num_way]
        permuted_labels.append(support_label[clsmap])
        permuted_ids[i, :].copy_(basis_matrix + clsmap.repeat(num_shot))
    return permuted_ids, permuted_labels


def one_hot(indices, depth):
    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size() + torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1, index, 1)
    return encoded_indicies


class MYNET(Net):
    def __init__(self, args, mode=None):
        super().__init__(args, mode)
        self.seminorm = False
        
        # 1. 先进行替换
        self._replace_wide_convs(self.encoder)
        
        # 2. 必须先将模型放到 GPU 上，再进行下面的推理维度探测
        self.encoder = self.encoder.cuda()
        
        # 3. 再探测维度
        dummy_input = torch.zeros(1, 1, 24000).cuda()
        with torch.no_grad():
            dummy_output = self.encoder(dummy_input)
            
        # 获取输出的通道数 (Channel 维度)，即 embed_dim
        embed_dim = dummy_output.shape[1]
        print(f"✅ 系统自动探测到的特征维度 (embed_dim): {embed_dim}")
        
        # 使用探测到的维度初始化 SPAMC
        self.spamc = SPAMC(embed_dim=embed_dim).cuda()

    # def _replace_wide_convs(self, module):
    #     for name, child in module.named_children():
    #         k_size = child.kernel_size[0] if isinstance(getattr(child, 'kernel_size', None), tuple) else getattr(child,
    #                                                                                                              'kernel_size',
    #                                                                                                              0)
    #         if isinstance(child, nn.Conv1d) and k_size == 16:
    #             refconv = ECA_RefConv1d(
    #                 in_channels=child.in_channels, out_channels=child.out_channels,
    #                 kernel_size=k_size, stride=child.stride[0] if isinstance(child.stride, tuple) else child.stride,
    #                 padding=child.padding[0] if isinstance(child.padding, tuple) else child.padding, groups=child.groups
    #             )
    #             refconv.weight.copy_(child.weight.data)
    #             if child.bias is not None:
    #                 refconv.bias = nn.Parameter(child.bias.data.clone())
    #             setattr(module, name, refconv)
    #         else:
    #             self._replace_wide_convs(child)

    def _replace_wide_convs(self, module):
        for name, child in module.named_children():
            if isinstance(child, nn.Conv1d):
                k_size = child.kernel_size[0] if isinstance(child.kernel_size, tuple) else child.kernel_size
                
                # 🚀 极其精准的打击：只替换那三个感受野为 8 的宽卷积！
                if k_size == 8:
                    refconv = ECA_RefConv1d(
                        in_channels=child.in_channels, out_channels=child.out_channels,
                        kernel_size=k_size, stride=child.stride[0] if isinstance(child.stride, tuple) else child.stride,
                        padding=child.padding[0] if isinstance(child.padding, tuple) else child.padding, groups=child.groups
                    )
                    refconv.weight.copy_(child.weight.data)
                    if child.bias is not None:
                        refconv.bias = nn.Parameter(child.bias.data.clone())
                    setattr(module, name, refconv)
                    # 打印确认
                    print(f"✅ 精准注入 ECA-RefConv 到宽卷积层: {name} (感受野={k_size})")
            else:
                self._replace_wide_convs(child)


    def split_instances(self, support_label, epoch):
        # 🚀 终极修复：不再强制全盘覆盖，而是模拟 2 个增量新类，恢复新旧知识碰撞！
        self.current_way = self.args.meta_new_class
        permuted_ids, permuted_labels = sample_task_ids(support_label, self.args.num_tasks, num_shot=self.args.sample_shot,
                                                        num_way=self.current_way, num_class=self.args.sample_class)
        return (permuted_ids.view(self.args.num_tasks, self.args.sample_shot, self.current_way), torch.stack(permuted_labels))

    def forward(self, x_shot, x_query=None, shot_label=None, epoch=None):
        if self.mode == 'encoder':
            return self.encode(x_shot)
        else:
            return self._forward(self.encode(x_shot), self.encode(x_query), self.split_instances(shot_label, epoch))

    def _forward(self, support, query, index_label):
        support_idx, support_labels = index_label
        num_task = support_idx.shape[0]
        support = support[support_idx.view(-1)].view(*(support_idx.shape + (-1,)))
        proto = support.mean(dim=1)

        num_proto = self.args.num_classes
        query = query.unsqueeze(1)

        # 🚀 在进入循环前，由于 query 数据刚刚通过了骨干网络前向传播，
        # 此时 self.encoder.gmpfm.last_lambda 中保存的正是当前 batch 查询样本的 λ 权重。
        # 它的形状为 [B, 3]，我们通过 unsqueeze(1) 将其调整为 [B, 1, 3] 备用。
        query_lambda = self.encoder.gmpfm.last_lambda.unsqueeze(1)

        logit = []
        for tt in range(num_task):
            global_mask = torch.eye(num_proto).cuda()
            whole_support_index = support_labels[tt, :]
            global_mask[:, whole_support_index] = 0
            local_mask = one_hot(whole_support_index, num_proto)

            current_classifier = torch.mm(self.fc.weight.t(), global_mask) + torch.mm(proto[tt, :].t(), local_mask)
            current_classifier = current_classifier.t().unsqueeze(0).expand(query.shape[0], num_proto, support.size(-1))

            # =========================================================
            # 🚀 SPAMC 尺度-语义双流协同校准 (通过真实提取路径完美打通)
            # =========================================================
            
            # 1. 构造语义上下文集合 C^(m) 
            C_m = torch.cat([current_classifier, query], dim=1) 
            
            # 2. 构造尺度先验集合 P^(m)
            # 2.1 从模型自带的 gate_memory 中依次取出已知老类别的平均门控权重 \overline{\lambda}
            known_lambdas = []
            for i in range(num_proto):
                if hasattr(self, 'gate_memory') and i in self.gate_memory:
                    known_lambdas.append(self.gate_memory[i].cuda())
                else:
                    # 容错处理：若还未存入记忆库（如 base session），使用均匀分布 (1/3, 1/3, 1/3) 占位
                    known_lambdas.append(torch.ones(3).cuda() / 3.0) 
            
            # 将老类别的 λ 堆叠并扩展至当前 batch 维度，形状变为 [B, num_proto, 3]
            known_lambdas = torch.stack(known_lambdas).unsqueeze(0).expand(query.shape[0], -1, -1)
            
            # 2.2 将老类别的先验与上面提取到的 query_lambda 进行拼接，得到完整的 P^(m) [B, num_proto + 1, 3]
            real_lambda_prior = torch.cat([known_lambdas, query_lambda], dim=1)
            
            # 3. 将语义上下文 C_m 和 尺度先验 real_lambda_prior 共同送入 SPAMC 模块校准
            C_calib = self.spamc(C_m, real_lambda_prior) 
            
            # 4. 拆分校准后的特征
            calib_classifier = C_calib[:, :-1, :] # [B, num_proto, feature_dim]
            calib_query = C_calib[:, -1:, :]      # [B, 1, feature_dim]

            # =========================================================
            
            # 正确使用温度系数乘法计算最终相似度余弦对齐评分
            logits = F.cosine_similarity(calib_query, calib_classifier, dim=-1)
            logits = logits * self.args.temperature
            logit.append(logits)

        logit = torch.cat(logit, 1)
        return logit.view(-1, self.args.num_classes)

    def updateclf(self, data, label):
        support_embs = self.encode(data)
        proto = support_embs.reshape(5, -1, support_embs.shape[-1]).mean(dim=0)
        self.fc.weight.data[torch.min(label):torch.max(label) + 1] = proto

    def forward_many(self, query):
        num_proto = self.args.num_classes
        # query 形状对齐为 [B, 1, Embed_dim]
        query = query.view(-1, 1, query.size(-1))
        current_classifier = self.fc.weight.unsqueeze(0).expand(query.shape[0], num_proto, query.size(-1))

        # =========================================================
        # 🚀 在测试阶段 (forward_many) 注入 SPAMC 协同校准
        # =========================================================
        
        # 1. 构造语义上下文集合 C^(m) 
        C_m = torch.cat([current_classifier, query], dim=1) 
        
        # 2. 构造尺度先验集合 P^(m)
        # 获取当前 query 的 λ (前向传播时刚被 encoder 存下)
        query_lambda = self.encoder.gmpfm.last_lambda.unsqueeze(1)
        
        # 获取已知老类别的 \overline{\lambda}
        known_lambdas = []
        for i in range(num_proto):
            if hasattr(self, 'gate_memory') and i in self.gate_memory:
                known_lambdas.append(self.gate_memory[i].cuda())
            else:
                known_lambdas.append(torch.ones(3).cuda() / 3.0) 
        
        known_lambdas = torch.stack(known_lambdas).unsqueeze(0).expand(query.shape[0], -1, -1)
        real_lambda_prior = torch.cat([known_lambdas, query_lambda], dim=1)
        
        # 3. 执行协同校准
        C_calib = self.spamc(C_m, real_lambda_prior) 
        calib_classifier = C_calib[:, :-1, :] # 校准后的分类器 [B, num_proto, feature_dim]
        calib_query = C_calib[:, -1:, :]      # 校准后的查询特征 [B, 1, feature_dim]

        # =========================================================

        # 🚀 核心修复 3：正确测试环境匹配，使用校准后的特征计算余弦相似度
        logits = F.cosine_similarity(calib_query, calib_classifier, dim=-1)
        logits = logits * self.args.temperature
        return logits
    

    # 在 MYNET 类中新增/重写此方法
    def update_fc(self, dataloader, class_list, session):
        self.eval()
        new_lambda_dict = {}
        
        with torch.no_grad():
            for class_id in class_list:
                features_list = []
                lambdas_list = []
                
                for batch in dataloader:
                    data, label = [_.cuda() for _ in batch]
                    # 筛选出属于当前类别 class_id 的样本
                    mask = (label == class_id)
                    if not mask.any():
                        continue
                    
                    class_data = data[mask]
                    
                    # 1. 前向传播提取特征（此时底层 gmpfm 模块内部会自动更新 last_lambda 属性）
                    if hasattr(self, 'encode'):
                        features = self.encode(class_data)
                    else:
                        features = self.encoder(class_data)
                        
                    features_list.append(features)
                    
                    # 🚀 核心修改：利用你确认的路径，实时提取当前 batch 产生的门控向量 λ
                    current_lambdas = self.encoder.gmpfm.last_lambda 
                    lambdas_list.append(current_lambdas)
                
                # 2. 计算当前类别的特征原型 (Prototype) 并更新分类器
                features_tensor = torch.cat(features_list, dim=0)
                prototype = features_tensor.mean(dim=0)
                self.fc.weight.data[class_id] = prototype
                
                # 🚀 核心修改：计算当前类别所有样本的平均门控向量 (Mean λ)，存入字典
                lambdas_tensor = torch.cat(lambdas_list, dim=0)
                mean_lambda = lambdas_tensor.mean(dim=0)
                new_lambda_dict[class_id] = mean_lambda.detach().cpu()
                
        # 返回新类别的门控权重字典，供 fscil_trainer.py 更新全局记忆库
        return new_lambda_dict