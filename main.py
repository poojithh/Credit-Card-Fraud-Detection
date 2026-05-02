from src.visualization import plot_feature_importance
from src.evaluation import evaluate_model
from src.supervised_models import train_models
from src.preprocessing import load_data, preprocess
from src.optuna_tuning import tune_xgboost
import torch
from src.autoencoder import Autoencoder


# Load and preprocess data
df = load_data("data/raw/creditcard.csv")
X_train, X_test, y_train, y_test = preprocess(df)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# Convert X_train to tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)

# Convert y_train to NumPy array first
y_train_array = y_train.to_numpy()

# Train only on NON-FRAUD transactions
X_train_normal = X_train_tensor[y_train_array == 0]

print("Normal transactions used for autoencoder:", X_train_normal.shape)

# Build Autoencoder model
model = Autoencoder(X_train.shape[1])
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(5):
    output = model(X_train_normal)
    loss = criterion(output, X_train_normal)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item():.6f}")


print("Running Optuna tuning for XGBoost...")
best_params = tune_xgboost(X_train, y_train)

print("Training XGBoost and Bagging models...")
xgb, bag = train_models(X_train, y_train, best_params)

print("Models trained successfully")

print("Evaluating XGBoost...")
evaluate_model(xgb, X_test, y_test, "XGBoost")

print("Evaluating Bagging Classifier...")
evaluate_model(bag, X_test, y_test, "Bagging Classifier")

plot_feature_importance(xgb)