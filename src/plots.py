import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


def plot_permutation_importance(model, X_val, y_val, feature_names: list, top_n: int = 10, output_path: str = "plots/feature_importance.png") -> None:
    """
    Computes and plots permutation feature importance for a fitted pipeline.
    Saves the figure to output_path.

    :param model: fitted sklearn/imblearn Pipeline
    :param X_val: validation features (DataFrame)
    :param y_val: validation target (Series)
    :param feature_names: list of feature names in display order
    :param output_path: path to save the PNG
    """
    result = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=42, scoring="roc_auc")

    # Sort by mean importance descending, keep top_n
    sorted_idx = result.importances_mean.argsort()[-top_n:]
    names = np.array(feature_names)[sorted_idx]
    means = result.importances_mean[sorted_idx]
    stds  = result.importances_std[sorted_idx]

    fig, ax = plt.subplots(figsize=(8, len(feature_names) * 0.4 + 1))
    ax.barh(names, means, xerr=stds, color="steelblue", ecolor="gray", capsize=3)
    ax.set_xlabel("Mean decrease in ROC-AUC")
    ax.set_title("Permutation Feature Importance (Random Forest)")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Feature importance plot saved to {output_path}")
