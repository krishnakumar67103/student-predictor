import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load data (separator ; or , auto-detect)
df = pd.read_csv("student-mat.csv", sep=None, engine="python")
print("Shape:", df.shape)
print(df.head())

# 2. Target: Pass (G3 >= 10) / Fail
df["result"] = (df["G3"] >= 10).astype(int)
df = df.drop(columns=["G3"])

# 3. Encode categorical columns
df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=["result"])
y = df["result"]

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Train models
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

best_model, best_acc = None, 0
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"\n{name} Accuracy: {acc:.3f}")
    print(classification_report(y_test, pred))
    if acc > best_acc:
        best_model, best_acc = model, acc

# 6. Save best model + column names
joblib.dump(best_model, "model.pkl")
joblib.dump(list(X.columns), "columns.pkl")
print("\nSaved model.pkl and columns.pkl")

import matplotlib.pyplot as plt
import pandas as pd

rf = models["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)[:10]
imp.plot(kind="barh", title="Top 10 Important Features")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()