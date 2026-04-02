"""
混淆矩阵准确率比较脚本
比较两个混淆矩阵的整体准确率和平均每类准确率
"""

import numpy as np
import os

def calculate_metrics(cm):
    """
    计算混淆矩阵的准确率指标
    
    Args:
        cm: 混淆矩阵 (numpy array)
    
    Returns:
        overall_acc: 整体准确率（对角线之和/总样本数）
        mean_per_class_acc: 平均每类准确率（每类对角线/该类总数，再取平均）
    """
    # 整体准确率：对角线之和除以总样本数
    overall_acc = np.trace(cm) / cm.sum()
    
    # 平均每类准确率：每行的对角线元素除以该行的总和，然后取平均
    per_class_acc = np.diag(cm) / (cm.sum(axis=1) + 1e-12)  # 加小值避免除零
    mean_per_class_acc = per_class_acc.mean()
    
    return overall_acc, mean_per_class_acc

def main():
    """主函数"""
    print("=" * 80)
    print("混淆矩阵准确率比较工具")
    print("=" * 80)
    
    # 混淆矩阵文件路径
    base_dir = 'results/confusion_matrices'
    myfix_npy = os.path.join(base_dir, 'myfix.npy')
    infogcn_npy = os.path.join(base_dir, 'infogcn.npy')
    
    # 检查文件是否存在
    if not os.path.exists(myfix_npy):
        print(f"错误：找不到文件 {myfix_npy}")
        return
    
    if not os.path.exists(infogcn_npy):
        print(f"错误：找不到文件 {infogcn_npy}")
        return
    
    # 加载混淆矩阵
    print(f"\n正在加载混淆矩阵...")
    cm_myfix = np.load(myfix_npy)
    cm_infogcn = np.load(infogcn_npy)
    
    print(f"myfix.npy 形状: {cm_myfix.shape}")
    print(f"infogcn.npy 形状: {cm_infogcn.shape}")
    
    # 计算指标
    print(f"\n正在计算准确率指标...")
    overall_myfix, mean_per_class_myfix = calculate_metrics(cm_myfix)
    overall_infogcn, mean_per_class_infogcn = calculate_metrics(cm_infogcn)
    
    # 输出结果
    print("\n" + "=" * 80)
    print("准确率比较结果")
    print("=" * 80)
    print(f"\n{'指标':<20} {'myfix.png':<20} {'infogcn.png':<20} {'差异':<15}")
    print("-" * 80)
    print(f"{'整体准确率':<20} {overall_myfix*100:>18.3f}% {overall_infogcn*100:>18.3f}% {overall_myfix-overall_infogcn:>13.3f}%")
    print(f"{'平均每类准确率':<20} {mean_per_class_myfix*100:>18.3f}% {mean_per_class_infogcn*100:>18.3f}% {mean_per_class_myfix-mean_per_class_infogcn:>13.3f}%")
    print("=" * 80)
    
    # 判断哪个准确率更高
    print("\n结论：")
    print("-" * 80)
    
    if overall_myfix > overall_infogcn:
        diff_overall = (overall_myfix - overall_infogcn) * 100
        print(f"✓ myfix.png 的整体准确率更高，高出 {diff_overall:.3f}%")
    elif overall_infogcn > overall_myfix:
        diff_overall = (overall_infogcn - overall_myfix) * 100
        print(f"✓ infogcn.png 的整体准确率更高，高出 {diff_overall:.3f}%")
    else:
        print("✓ 两个混淆矩阵的整体准确率相同")
    
    if mean_per_class_myfix > mean_per_class_infogcn:
        diff_mean = (mean_per_class_myfix - mean_per_class_infogcn) * 100
        print(f"✓ myfix.png 的平均每类准确率更高，高出 {diff_mean:.3f}%")
    elif mean_per_class_infogcn > mean_per_class_myfix:
        diff_mean = (mean_per_class_infogcn - mean_per_class_myfix) * 100
        print(f"✓ infogcn.png 的平均每类准确率更高，高出 {diff_mean:.3f}%")
    else:
        print("✓ 两个混淆矩阵的平均每类准确率相同")
    
    print("=" * 80)
    
    # 额外统计信息
    print("\n额外统计信息：")
    print("-" * 80)
    print(f"myfix.png:")
    print(f"  - 总样本数: {int(cm_myfix.sum())}")
    print(f"  - 正确预测数: {int(np.trace(cm_myfix))}")
    print(f"  - 错误预测数: {int(cm_myfix.sum() - np.trace(cm_myfix))}")
    
    print(f"\ninfogcn.png:")
    print(f"  - 总样本数: {int(cm_infogcn.sum())}")
    print(f"  - 正确预测数: {int(np.trace(cm_infogcn))}")
    print(f"  - 错误预测数: {int(cm_infogcn.sum() - np.trace(cm_infogcn))}")
    print("=" * 80)

if __name__ == '__main__':
    main()

