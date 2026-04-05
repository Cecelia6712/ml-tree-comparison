决策树与随机森林综合对比实验



项目简介

独立实现ID3、CART决策树（分类）及CART回归树，在Iris、Digits、Wine、Diabetes数据集上与逻辑回归、感知机进行对比，并实现回归树的剪枝优化。



实验内容

手写ID3（信息增益+连续值二分法）和CART（基尼系数）分类树

对比逻辑回归、感知机在多个数据集上的Precision、Recall、F1、AUC

回归任务：CART回归树 + 代价复杂度剪枝（预剪枝/后剪枝）

决策树可视化（graphviz）



运行方法

1\. 安装依赖：`pip install numpy pandas scikit-learn matplotlib graphviz pydotplus`

2\. 安装graphviz系统软件（https://graphviz.org/download/）

3\. 运行：`python run\_all.py`



结果示例

分类对比表格和回归剪枝前后MSE对比见终端输出，决策树图保存在`results/figures/`。



技术栈

Python, NumPy, scikit-learn, Matplotlib, Graphviz

