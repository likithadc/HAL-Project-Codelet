# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE

# Load dataset from CSV file
credit_card_data = pd.read_csv(r'C:\Users\WELCOME\Downloads\creditcard.csv')  # Correct file path

# Display dataset information
print("Dataset Loaded: Rows:", credit_card_data.shape[0], "Columns:", credit_card_data.shape[1])
print(credit_card_data.head())

# Check for missing values
print("Missing values:", credit_card_data.isnull().sum())

# Separate legit and fraud transactions
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

print("Legit transactions:", legit.shape)
print("Fraud transactions:", fraud.shape)

# Describe amount statistics
print("Legit transactions amount stats:\n", legit.Amount.describe())
print("Fraud transactions amount stats:\n", fraud.Amount.describe())

# Feature Scaling (Standardization) for all features, not just 'Amount'
scaler = StandardScaler()
X = credit_card_data.drop(columns=['Class'])  # Features excluding the target column
Y = credit_card_data['Class']  # Target column

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=2)
X_resampled, Y_resampled = smote.fit_resample(X, Y)
print(f"Resampled dataset: X_resampled: {X_resampled.shape}, Y_resampled: {Y_resampled.shape}")

# Split dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_resampled, Y_resampled, test_size=0.2, stratify=Y_resampled, random_state=2)
print("Train/Test Split - X_train:", X_train.shape, "X_test:", X_test.shape)

# Feature Scaling for all features (not just 'Amount')
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled: X_train_scaled:", X_train_scaled.shape, "X_test_scaled:", X_test_scaled.shape)

# Train Logistic Regression Model with increased max_iter
lr_model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=2)
lr_model.fit(X_train_scaled, Y_train)
print("Logistic Regression Model Trained.")

# Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=50, random_state=2, n_jobs=-1, verbose=1)  
rf_model.fit(X_train_scaled, Y_train)
print("Random Forest Model Trained.")

# Model Predictions
lr_test_pred = lr_model.predict(X_test_scaled)
rf_test_pred = rf_model.predict(X_test_scaled)

# Function to Evaluate Model Performance
def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"\n{model_name} Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

# Evaluate Logistic Regression
evaluate_model(Y_test, lr_test_pred, "Logistic Regression")

# Evaluate Random Forest
evaluate_model(Y_test, rf_test_pred, "Random Forest")

# Confusion Matrix for Random Forest
cm = confusion_matrix(Y_test, rf_test_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Random Forest Confusion Matrix')
plt.show()

# Feature Importance - Random Forest
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf_model.feature_importances_})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
print("Feature Importance:\n", feature_importance)

# ROC Curve - Random Forest
rf_probabilities = rf_model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(Y_test, rf_probabilities)
roc_auc = roc_auc_score(Y_test, rf_probabilities)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='red')  # Random guessing diagonal
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend()
plt.show()

# Cross-validation Score for Random Forest
cv_scores = cross_val_score(rf_model, X_resampled, Y_resampled, cv=5, scoring='accuracy')
print(f'Cross-Validation Accuracy: {cv_scores.mean():.4f}')