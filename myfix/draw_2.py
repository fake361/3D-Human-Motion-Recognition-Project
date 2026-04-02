# import pickle
from model.mytest import MyfixGCN
import numpy as np
import torch
from collections import OrderedDict
import matplotlib.pyplot as plt
import os


def main():

    # Windows兼容性：使用相对路径替代硬编码的Linux路径
    weights = os.path.join('results', 'ntu_NTU60_CS_fix1_k8', 'runs-102-31926.pt')

    load_weights = torch.load(weights)
    # print(load_weights['l9.agcn.shared_topology'])

    tensor = load_weights['l9.agcn.shared_topology']  # 随机生成一个示例张量

    for i in range(tensor.shape[0]):
        # 获取第i个矩阵的对角线元素
        diagonal = torch.diagonal(tensor[i])
        print(f"第 {i + 1} 个矩阵的对角线元素: {diagonal.numpy()}")

if __name__ == '__main__':
    main()