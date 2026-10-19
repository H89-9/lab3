import numpy as np
from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    X_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=float).ravel()
    X_test = np.asarray(data.test_features, dtype=float)

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    n_samples, n_features = X_train.shape
    weights = np.zeros(n_features)
    bias = 0.0
    loss_history = []

    for _ in range(data.epochs):
        predicted = X_train @ weights + bias
        errors = predicted - y_train

        mse = np.mean(errors ** 2)
        loss_history.append(float(mse))

        grad_weights = (2 / n_samples) * (X_train.T @ errors)
        grad_bias = 2 * np.mean(errors)

        weights -= data.learning_rate * grad_weights
        bias -= data.learning_rate * grad_bias

    predictions = X_test @ weights + bias

    return RegressionResult(
        weights=weights,
        bias=float(bias),
        loss_history=loss_history,
        predictions=predictions,
    )


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    X_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=float).ravel()
    X_test = np.asarray(data.test_features, dtype=float)

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    n_samples, n_features = X_train.shape

    weights = np.zeros(n_features)
    bias = 0.0
    loss_history = []

    def sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    for _ in range(data.epochs):
        z = X_train @ weights + bias
        probabilities = sigmoid(z)

        p = np.clip(probabilities, 1e-12, 1 - 1e-12)

        loss = -np.mean(
            y_train * np.log(p)
            + (1 - y_train) * np.log(1 - p)
        )
        loss_history.append(float(loss))

        errors = probabilities - y_train

        grad_weights = (X_train.T @ errors) / n_samples
        grad_bias = np.mean(errors)

        weights -= data.learning_rate * grad_weights
        bias -= data.learning_rate * grad_bias

    test_probabilities = sigmoid(X_test @ weights + bias)
    predictions = (test_probabilities >= 0.5).astype(int)

    return ClassificationResult(
        weights=weights,
        bias=float(bias),
        loss_history=loss_history,
        probabilities=test_probabilities,
        predictions=predictions,
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score
    )
    from grader_contracts.linear_models import ClassificationMetrics

    X_train = np.asarray(data.train_features, dtype=float)
    X_test = np.asarray(data.test_features, dtype=float)
    y_train = np.asarray(data.train_target).ravel()
    y_test = np.asarray(data.test_target).ravel()

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        ),
        "knn": KNeighborsClassifier(
            n_neighbors=5
        )
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        metrics = ClassificationMetrics(
            roc_auc=float(roc_auc_score(y_test, probabilities)),
            accuracy=float(accuracy_score(y_test, predictions)),
            precision=float(precision_score(y_test, predictions)),
            recall=float(recall_score(y_test, predictions)),
            f1=float(f1_score(y_test, predictions))
        )

        results[name] = metrics

    return BaselineComparisonResult(
        logistic_regression=results["logistic_regression"],
        decision_tree=results["decision_tree"],
        knn=results["knn"]
    )
