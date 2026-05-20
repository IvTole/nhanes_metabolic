from datetime import datetime
from src.io import Dataset
from src.evaluation import ModelEvaluation
from src.preprocessing import build_preprocessor
from src.plots import (
    plot_permutation_importance,
    plot_roc_curves,
    plot_confusion_matrix,
    plot_metrics_heatmap,
)

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


def build_pipeline(model):
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("smote", SMOTE(random_state=42)),
        ("model", model)
    ])


def main(plot: bool = True):
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("Start Date and Time: ", start_datetime)

    data = Dataset()
    X_train, y_train = data.load_data_xy()
    print(f"X_train shape: {X_train.shape}")

    models = {
        "lr":  LogisticRegression(
                   solver="lbfgs", max_iter=5000,
                   C=0.1,                      # regularización moderada
                   class_weight="balanced",
               ),
        "rf":  RandomForestClassifier(
                   n_estimators=200,
                   max_depth=10,
                   min_samples_leaf=5,
                   class_weight="balanced",
                   random_state=42,
               ),
        "gb":  GradientBoostingClassifier(
                   n_estimators=200,
                   learning_rate=0.05,
                   max_depth=4,
                   subsample=0.8,
                   random_state=42,
               ),
        "svc": SVC(
                   C=1.0, kernel="rbf", gamma="scale",
                   probability=True, class_weight="balanced",
                   random_state=42,
               ),
        "mlp": MLPClassifier(
                   hidden_layer_sizes=(64, 32), activation="relu",
                   alpha=0.01, early_stopping=True,
                   max_iter=1000, random_state=42,
               ),
        "knn": KNeighborsClassifier(
                   n_neighbors=11, weights="distance", metric="euclidean",
               ),
        "gnb": GaussianNB(
                   var_smoothing=1.0E-08,
               ),
    }

    cv_results = []
    fitted_pipelines = {}

    for tag, model in models.items():
        print(f"\n{'='*40}")
        pipeline = build_pipeline(model)
        ev = ModelEvaluation(X=X_train, y=y_train, tag=tag)
        cv_results.append(ev.evaluate_cv(pipeline))
        ev.evaluate_model(pipeline)
        fitted_pipelines[tag] = pipeline

    if plot:
        # Shared validation set from first ModelEvaluation — use best model's split
        ev_ref = ModelEvaluation(X=X_train, y=y_train, tag="plots", random_state=42)

        # Re-fit all pipelines on the same train split for fair comparison
        fitted_for_plots = {}
        for tag, model in models.items():
            p = build_pipeline(model)
            p.fit(ev_ref.X_train, ev_ref.y_train)
            fitted_for_plots[tag] = p

        plot_roc_curves(fitted_for_plots, ev_ref.X_valid, ev_ref.y_valid)
        plot_metrics_heatmap(cv_results)

        # Confusion matrix for best model by roc_auc
        # cv_results stores class name in "model" key; map back to short tag
        model_name_to_tag = {type(m).__name__: tag for tag, m in models.items()}
        best_model_name = max(cv_results, key=lambda r: r.get("roc_auc", 0))["model"]
        best_tag = model_name_to_tag[best_model_name]
        plot_confusion_matrix(fitted_for_plots[best_tag], ev_ref.X_valid, ev_ref.y_valid,
                              label=best_tag, output_path=f"plots/confusion_matrix_{best_tag}.png")

        # Permutation importance for RF
        plot_permutation_importance(fitted_for_plots["rf"], ev_ref.X_valid, ev_ref.y_valid,
                                    feature_names=ev_ref.X_valid.columns.tolist())


if __name__ == "__main__":
    main()
