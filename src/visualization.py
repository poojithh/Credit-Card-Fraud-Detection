import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importance(model, filename="results/plots/feature_importance.png"):
    importance = model.feature_importances_
    features = np.arange(len(importance))

    plt.figure(figsize=(10, 6))
    plt.bar(features, importance)
    plt.xlabel("Feature Index")
    plt.ylabel("Importance Score")
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    print(f"Feature importance plot saved to {filename}")