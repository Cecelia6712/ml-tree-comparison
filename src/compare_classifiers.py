import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_digits, load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import label_binarize
from decision_tree import ID3DecisionTree, CARTDecisionTree

def run_comparison():
    datasets = {
        "Iris": load_iris(),
        "Digits": load_digits(),
        "Wine": load_wine()
    }
    models = {
        "ID3 (our)": ID3DecisionTree(max_depth=5),
        "CART (our)": CARTDecisionTree(max_depth=5),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Perceptron": Perceptron(max_iter=1000)
    }
    results = []
    for dname, data in datasets.items():
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        for mname, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # 多分类 AUC 需要 predict_proba
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)
                auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
            else:
                auc = None
            prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            results.append({
                "Dataset": dname,
                "Model": mname,
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1": round(f1, 4),
                "AUC": round(auc, 4) if auc else None
            })
    df = pd.DataFrame(results)
    print(df)
    df.to_csv("results/comparison_results.csv", index=False)
    print("\n结果已保存至 results/comparison_results.csv")

if __name__ == "__main__":
    run_comparison()