import torch
from models.base.Network import MYNET
import argparse

# 模拟 args
args = argparse.Namespace(
    dataset='mbhm',
    num_classes=10,
    base_class=5,
    way=5,
    temperature=16
)

# 实例化模型
model = MYNET(args, mode='encoder').cuda()
model.eval()

# 模拟一个 batch 的 1D 信号输入
dummy_input = torch.randn(2, 1, 1024).cuda()

with torch.no_grad():
    output = model(dummy_input)
    print(f"模型输出维度: {output.shape}") 
    # 预期输出应该是 (2, 128)，如果看到这个结果，说明网络完全正常工作！