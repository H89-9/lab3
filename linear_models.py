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
    """Сравните готовые Logistic Regression, Decision Tree и KNN на одном split.

    После запуска запишите реальные результаты в experiment_results.json:
    {"linear_regression":{"test_mse": число},
     "classification":{"own_logistic": {"roc_auc": ..., "accuracy": ..., "precision": ..., "recall": ..., "f1": ...},
                       "logistic_regression": {...}, "decision_tree": {...}, "knn": {...}}}.
    """
    raise NotImplementedError
