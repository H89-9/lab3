import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from linear_models import train_linear_regression
from grader_contracts.linear_models import RegressionInput

df = pd.read_csv("Student_Performance.csv")

df["Extracurricular Activities"] = df[
    "Extracurricular Activities"
].map({"Yes": 1, "No": 0})

X = df.drop(columns=["Performance Index"])
y = df["Performance Index"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

data = RegressionInput(
    train_features=X_train,
    train_target=y_train,
    test_features=X_test,
    test_target=y_test,
    learning_rate=0.05,
    epochs=800
)

result = train_linear_regression(data)

mse = mean_squared_error(y_test, result.predictions)

print("Количество строк:", len(df))
print("Начальная ошибка:", result.loss_history[0])
print("Конечная ошибка:", result.loss_history[-1])
print("MSE на тестовой выборке:", mse)

print("\nПервые 5 прогнозов:")
for actual, predicted in zip(
    y_test.iloc[:5], result.predictions[:5]
):
    print(f"Реальное: {actual:.1f}, прогноз: {predicted:.1f}")