import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("/content/Fraud Detection Dataset.csv")

df.drop(["Transaction_ID", "User_ID"], axis=1, inplace=True)

categorical_columns = [
    "Transaction_Type",
    "Device_Used",
    "Location",
    "Payment_Method"
]

encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df.drop("Fraudulent", axis=1)
y = df["Fraudulent"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("=" * 40)
print("      FRAUD DETECTION RESULTS")
print("=" * 40)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2%}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

sample_transaction = pd.DataFrame({
    "Transaction_Amount": [2500],
    "Transaction_Type": [2],
    "Time_of_Transaction": [14],
    "Device_Used": [1],
    "Location": [3],
    "Previous_Fraudulent_Transactions": [0],
    "Account_Age": [24],
    "Number_of_Transactions_Last_24H": [5],
    "Payment_Method": [1]
})

prediction = model.predict(sample_transaction)

print("\nSample Transaction Prediction:")
if prediction[0] == 1:
    print("Fraudulent Transaction")
else:
    print("Legitimate Transaction")
