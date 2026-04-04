import subprocess
import os
import sys

# 添加 src 目录到 Python 路径
sys.path.append('src')

if __name__ == "__main__":
    print("="*50)
    print("运行分类对比实验...")
    subprocess.run([sys.executable, "src/compare_classifiers.py"])
    print("\n" + "="*50)
    print("运行回归剪枝实验...")
    subprocess.run([sys.executable, "src/regression_tree.py"])
    print("\n全部实验完成！结果保存在 results/ 文件夹下。")