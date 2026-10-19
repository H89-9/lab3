import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from linear_models import train_logistic_classifier
from grader_contracts.linear_models import ClassificationInput

df = pd.read_csv("german.csv", sep=";")

X = df.drop(columns=["Creditability"])
y = df["Creditability"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

data = ClassificationInput(
    train_features=X_train,
    train_target=y_train,
    test_features=X_test,
    test_target=y_test,
    learning_rate=0.1,
    epochs=1000
)

result = train_logistic_classifier(data)

accuracy = accuracy_score(y_test, result.predictions)
precision = precision_score(y_test, result.predictions)
recall = recall_score(y_test, result.predictions)
f1 = f1_score(y_test, result.predictions)
roc_auc = roc_auc_score(y_test, result.probabilities)

print("Количество клиентов:", len(df))
print("Начальная ошибка:", result.loss_history[0])
print("Конечная ошибка:", result.loss_history[-1])

print("\nМетрики модели:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)
print("ROC-AUC:", roc_auc)

print("\nПервые 5 прогнозов:")
for actual, predicted, probability in zip(
    y_test.iloc[:5],
    result.predictions[:5],
    result.probabilities[:5]
):
    print(
        f"Реальное: {actual}, "
        f"Прогноз: {predicted}, "
        f"Вероятность класса 1: {probability:.2f}"
    )