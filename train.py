from datetime import datetime
from src.io import Dataset
from src.evaluation import ModelEvaluation
from src.preprocessing import build_preprocessor
from src.plots import plot_permutation_importance
from src.config import FEATURES

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def main(plot: bool = True):
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("Start Date and Time: ", start_datetime)

    data = Dataset()
    X_train, y_train = data.load_data_xy()
    print(f"X_train shape: {X_train.shape}")

    ## Model pipelines

    # pipeline_lr = Pipeline([
    #     ("preprocessor", build_preprocessor()),
    #     ("smote", SMOTE(random_state=42)),
    #     ("model", LogisticRegression(solver="lbfgs", max_iter=5000))
    # ])

    pipeline_rf = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("smote", SMOTE(random_state=42)),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Evaluate
    ev = ModelEvaluation(X=X_train, y=y_train, tag='rf')
    ev.evaluate_cv(pipeline_rf)
    ev.evaluate_model(pipeline_rf)

    # Feature importance
    if plot:
        plot_permutation_importance(pipeline_rf, ev.X_valid, ev.y_valid, feature_names=ev.X_valid.columns.tolist())


if __name__ == "__main__":
    main()
