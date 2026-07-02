import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
df = pd.read_csv("heart.csv")
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values check:")
print(df.isnull().sum())

# Define Features and Target
X = df.drop(columns=['target'])
y = df['target']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Analyze Scaling impact
for scaled in [False, True]:
    print(f"\n--- Model Performance (Scaling = {scaled}) ---")
    if scaled:
        scaler = StandardScaler()
        X_train_processed = scaler.fit_transform(X_train)
        X_test_processed = scaler.transform(X_test)
    else:
        X_train_processed = X_train.values
        X_test_processed = X_test.values

    # Find best K
    k_range = range(1, 21)
    k_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train_processed, y_train, cv=5, scoring='accuracy')
        k_scores.append(scores.mean())
    
    best_k = k_range[np.argmax(k_scores)]
    print(f"Best K (based on 5-fold CV): {best_k} with Mean CV Accuracy = {k_scores[best_k-1]:.4f}")
    
    # Train and evaluate model with best K
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train_processed, y_train)
    y_pred = knn.predict(X_test_processed)
    
    print(f"Test Set Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
