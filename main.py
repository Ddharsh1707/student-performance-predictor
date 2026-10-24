"""Student Performance Predictor - predicts pass/fail with a Decision Tree and prints its rules."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text


def make_students(n=600, seed=1):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "study_hours": rng.uniform(0, 10, n).round(1),
        "attendance_pct": rng.uniform(40, 100, n).round(0),
        "previous_score": rng.uniform(30, 100, n).round(0),
        "sleep_hours": rng.uniform(4, 9, n).round(1),
    })
    score = (4 * df.study_hours + 0.4 * df.attendance_pct + 0.3 * df.previous_score
             + 2 * (df.sleep_hours > 6) + rng.normal(0, 5, n))
    df["passed"] = (score > 65).astype(int)
    return df


def main():
    df = make_students()
    print(f"{len(df)} students, pass rate {df.passed.mean():.0%}\n")

    X, y = df.drop(columns="passed"), df["passed"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    tree = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_train)
    pred = tree.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, pred):.2%}")
    print(classification_report(y_test, pred, target_names=["fail", "pass"]))

    print("Rules the tree learned:")
    print(export_text(tree, feature_names=list(X.columns)))

    student = pd.DataFrame([{"study_hours": 3, "attendance_pct": 85, "previous_score": 70, "sleep_hours": 7}])
    chance = tree.predict_proba(student)[0][1]
    print(f"New student {student.iloc[0].to_dict()} -> chance of passing {chance:.0%}")


if __name__ == "__main__":
    main()
