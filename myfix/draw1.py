
# import pickle
from model.mytest import MyfixGCN

import torch
from collections import OrderedDict
import matplotlib.pyplot as plt
import os


def main():

    # Windows兼容性：使用相对路径替代硬编码的Linux路径
    weights = os.path.join('results', 'ntu_NTU60_CS_fix1_k8', 'runs-108-33804.pt')

    load_weights = torch.load(weights)
    # print(load_weights['l9.agcn.shared_topology'])

    tensor = load_weights['l9.agcn.shared_topology']  # 随机生成一个示例张量
    # 将张量拆分为 3 个 (25, 25) 的矩阵
    duis=[]
    removes=[[1, 4, 8,11,15,18],[2,9,14,16,18,20],[1,3,7,12,15,18]]


    for i in range(tensor.shape[0]):
        # 获取第i个矩阵的对角线元素
        diagonal = torch.diagonal(tensor[i])
        indices_to_keep = [j for j in range(25) if j not in removes[i]]
        result_tensor = diagonal[indices_to_keep]
        duis.append(result_tensor)

    weights = os.path.join('results', 'ntu_NTU60_CS_fix1_k8', 'runs-102-31926.pt')

    load_weights = torch.load(weights)
    # print(load_weights['l9.agcn.shared_topology'])

    tensor = load_weights['l9.agcn.shared_topology']  # 随机生成一个示例张量
    matrices = [tensor[i] for i in range(3)]  # 拆分为 3 个矩阵
    new=[]

    matrix=matrices[0]
    # 要删除的行和列的索引（假设删除第 3、7、11、15、19、23 行和列）
    rows_to_remove = [2, 6, 10, 14, 18, 22]  # Python 索引从 0 开始
    cols_to_remove = [1, 4, 8,11,15,18]

    row_mask = torch.ones(25, dtype=torch.bool)  # 创建一个全为 True 的掩码
    row_mask[rows_to_remove] = False  # 将需要删除的行标记为 False
    matrix = matrix[row_mask]  # 使用掩码删除行

    # 删除指定的列
    col_mask = torch.ones(25, dtype=torch.bool)  # 创建一个全为 True 的掩码
    col_mask[cols_to_remove] = False  # 将需要删除的列标记为 False
    new_matrix = matrix[:, col_mask]  # 使用掩码删除列
    # 使用切片操作删除指定的行和列

    new.append(new_matrix)

    # 假设有一个形状为 (3, 25, 25) 的张量
    matrix = matrices[1]
    # 要删除的行和列的索引（假设删除第 3、7、11、15、19、23 行和列）
    rows_to_remove = [2, 6, 10, 14, 18, 22]  # Python 索引从 0 开始
    cols_to_remove = [2,9,14,16,18,20]

    # 使用切片操作删除指定的行和列
    row_mask = torch.ones(25, dtype=torch.bool)  # 创建一个全为 True 的掩码
    row_mask[rows_to_remove] = False  # 将需要删除的行标记为 False
    matrix = matrix[row_mask]  # 使用掩码删除行

    # 删除指定的列
    col_mask = torch.ones(25, dtype=torch.bool)  # 创建一个全为 True 的掩码
    col_mask[cols_to_remove] = False  # 将需要删除的列标记为 False
    new_matrix = matrix[:, col_mask]  # 使用掩码删除列

    new.append(new_matrix)

    matrix = matrices[2]
    # 要删除的行和列的索引（假设删除第 3、7、11、15、19、23 行和列）
    rows_to_remove = [2, 6, 10, 14, 18, 22]  # Python 索引从 0 开始
    cols_to_remove = [1,3,7,12,15,18]

    # 使用切片操作删除指定的行和列
    row_mask = torch.ones(25, dtype=torch.bool)  # 创建一个全为 True 的掩码
    row_mask[rows_to_remove] = False  # 将需要删除的行标记为 False
    matrix = matrix[row_mask]  # 使用掩码删除行

    # 删除指定的列
    col_mask = torch.ones(25, dtype=torch.bool)  # 创建一个全为 True 的掩码
    col_mask[cols_to_remove] = False  # 将需要删除的列标记为 False
    new_matrix = matrix[:, col_mask]  # 使用掩码删除列

    new.append(new_matrix)

    for i in range(len(new)):
        for j in range(19):
            new[i][j, j] = duis[i][j]


    # Windows兼容性：使用相对路径替代硬编码的Linux路径
    save_dir = os.path.join("results", "aaa_pics", "3.20")  # 保存图像的目录
    os.makedirs(save_dir, exist_ok=True)  # 创建目录（如果不存在）
    global_min = torch.min(torch.stack(matrices)).item()
    global_max = torch.max(torch.stack(matrices)).item()

    # 遍历每个矩阵，保存为单独的图像
    for i, matrix in enumerate(new):
        # 创建图像
        plt.figure(figsize=(5, 5))
        plt.imshow(matrix, cmap='Blues', vmin=global_min, vmax=global_max)

        # 添加标题和标注
        min_val = torch.min(matrix).item()
        max_val = torch.max(matrix).item()
        plt.title(f'Matrix {i + 1}\nMin: {min_val:.2f}, Max: {max_val:.2f}')
        plt.axis('off')  # 隐藏坐标轴

        # plt.colorbar()
        # 保存图像
        save_path = os.path.join(save_dir, f'matrix_{i + 1}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)  # 保存为 PNG 文件
        plt.close()  # 关闭当前图像，释放内存

    print(f"图像已保存到目录: {save_dir}")


    # tr_list=[]
    # for i in [1,2]:
    #     one_list=[]
    #     for idx in [1,2,3,4,5,6,7,8,9]:
    #         one_list.append(load_weights[f'l{idx}.tcn.tcn{i}.2.tr'].tolist())
    #     tr_list.append(one_list)
    #
    # print(tr_list[0])

if __name__ == '__main__':
    main()