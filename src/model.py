import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


def get_models(random_state: int = 16) -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=100, random_state=random_state, class_weight='balanced'
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=random_state, class_weight='balanced'
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
        "SVM": CalibratedClassifierCV(
            SVC(class_weight='balanced', random_state=random_state), ensemble=False
        ),
        "KNN": KNeighborsClassifier(),
    }


def train_and_evaluate(models: dict, X_train, y_train, X_test, y_test) -> tuple[dict, pd.DataFrame]:
    trained_models = {}
    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        results.append({
            "Model": name,
            "Accuracy": model.score(X_test, y_test),
            "Precision": report['1']['precision'],
            "Recall": report['1']['recall'],
            "F1-Score": report['1']['f1-score'],
        })

    return trained_models, pd.DataFrame(results)


def get_top_features(rf_model, X_train, n: int = 10) -> list:
    importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
    return importances.sort_values(ascending=False).head(n).index.tolist()


def save_artifacts(model, scaler, top_features: list, output_dir: str = "../models"):
    joblib.dump(model, f"{output_dir}/alzheimer_rf_model.pkl")
    joblib.dump(scaler, f"{output_dir}/scaler.pkl")
    joblib.dump(top_features, f"{output_dir}/top_features.pkl")