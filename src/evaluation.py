from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def evaluate_model(model, X_test, y_test, model_name):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, probs)
    report = classification_report(y_test, preds)
    matrix = confusion_matrix(y_test, preds)

    print(f"\n===== {model_name} Evaluation =====")
    print("ROC-AUC Score:", roc_auc)
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(matrix)

    with open("results/metrics.txt", "a") as f:
        f.write(f"\n===== {model_name} Evaluation =====\n")
        f.write(f"ROC-AUC Score: {roc_auc}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(matrix))
        f.write("\n")