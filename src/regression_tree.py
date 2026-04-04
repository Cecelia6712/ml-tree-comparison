from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

def run_regression():
    os.makedirs("results/figures", exist_ok=True)
    # 使用 diabetes 数据集（无需下载，避免 SSL 错误）
    data = load_diabetes()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 不剪枝的回归树（完全生长）
    reg_unpruned = DecisionTreeRegressor(max_depth=None, random_state=42)
    reg_unpruned.fit(X_train, y_train)
    mse_un = mean_squared_error(y_test, reg_unpruned.predict(X_test))

    # 剪枝：代价复杂度剪枝 (ccp_alpha)
    path = reg_unpruned.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas[:-1]  # 去掉最后一个无穷大
    # 选择一个合适的 alpha（倒数第二个，如果不够则用 0.01）
    alpha = ccp_alphas[-2] if len(ccp_alphas) > 1 else 0.01
    reg_pruned = DecisionTreeRegressor(ccp_alpha=alpha, random_state=42)
    reg_pruned.fit(X_train, y_train)
    mse_pr = mean_squared_error(y_test, reg_pruned.predict(X_test))

    print(f"Diabetes 数据集回归结果（代替 California Housing）:")
    print(f"未剪枝 MSE: {mse_un:.4f}")
    print(f"剪枝后 MSE: {mse_pr:.4f}")

    # 可视化（限制深度为3，否则树太大看不清）
    plt.figure(figsize=(20,10))
    plot_tree(reg_unpruned, max_depth=3, filled=True, feature_names=data.feature_names)
    plt.title("Unpruned Regression Tree (max_depth=3)")
    plt.savefig("results/figures/regression_unpruned.png", dpi=300)
    plt.close()

    plt.figure(figsize=(20,10))
    plot_tree(reg_pruned, max_depth=3, filled=True, feature_names=data.feature_names)
    plt.title("Pruned Regression Tree (max_depth=3)")
    plt.savefig("results/figures/regression_pruned.png", dpi=300)
    plt.close()
    print("决策树图已保存至 results/figures/")

if __name__ == "__main__":
    run_regression()