import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay, confusion_matrix
import pandas as pd


def plot_permutation_importance(model, X_val, y_val, feature_names: list, top_n: int = 10, output_path: str = "plots/feature_importance.png") -> None:
    """
    Computes and plots permutation feature importance for a fitted pipeline.
    Saves the figure to output_path.

    :param model: fitted sklearn/imblearn Pipeline
    :param X_val: validation features (DataFrame)
    :param y_val: validation target (Series)
    :param feature_names: list of feature names in display order
    :param top_n: number of top features to display
    :param output_path: path to save the PNG
    """
    result = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=42, scoring="roc_auc")

    sorted_idx = result.importances_mean.argsort()[-top_n:]
    names = np.array(feature_names)[sorted_idx]
    means = result.importances_mean[sorted_idx]
    stds  = result.importances_std[sorted_idx]

    fig, ax = plt.subplots(figsize=(8, top_n * 0.4 + 1))
    ax.barh(names, means, xerr=stds, color="steelblue", ecolor="gray", capsize=3)
    ax.set_xlabel("Mean decrease in ROC-AUC")
    ax.set_title("Permutation Feature Importance (Random Forest)")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_roc_curves(fitted_pipelines: dict, X_val, y_val, output_path: str = "plots/roc_curves.png") -> None:
    """
    Plots overlapping ROC curves for multiple fitted pipelines.

    :param fitted_pipelines: dict of {label: fitted_pipeline}
    :param X_val: validation features
    :param y_val: validation target
    :param output_path: path to save the PNG
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    for label, pipeline in fitted_pipelines.items():
        model = pipeline.named_steps["model"]
        if hasattr(model, "predict_proba"):
            y_score = pipeline.predict_proba(X_val)[:, 1]
        else:
            y_score = pipeline.decision_function(X_val)
        fpr, tpr, _ = roc_curve(y_val, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_confusion_matrix(pipeline, X_val, y_val, label: str = "", output_path: str = "plots/confusion_matrix.png") -> None:
    """
    Plots the confusion matrix for a fitted pipeline.

    :param pipeline: fitted sklearn/imblearn Pipeline
    :param X_val: validation features
    :param y_val: validation target
    :param label: model name for the title
    :param output_path: path to save the PNG
    """
    y_pred = pipeline.predict(X_val)
    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No MetS", "MetS"])

    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {label}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_metrics_heatmap(cv_results: list, output_path: str = "plots/metrics_heatmap.png") -> None:
    """
    Plots a heatmap of CV metrics across models.

    :param cv_results: list of dicts returned by evaluate_cv()
    :param output_path: path to save the PNG
    """
    df = pd.DataFrame(cv_results).set_index("model")

    fig, ax = plt.subplots(figsize=(len(df.columns) * 1.2 + 1, len(df) * 0.6 + 1))
    im = ax.imshow(df.values, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)

    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            ax.text(j, i, f"{df.values[i, j]:.3f}", ha="center", va="center", fontsize=9)

    ax.set_title("CV Metrics — Model Comparison")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")
