# Program 9 — K-Nearest Neighbours on Iris Dataset
# VTU 22CDL66 Machine Learning Lab
# Using scikit-learn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ── 1. Load Dataset ───────────────────────────────────────────────────────────
iris = load_iris()
X, y = iris.data, iris.target
class_names = iris.target_names

print("=" * 55)
print("   KNN Classifier — Iris Dataset (VTU 22CDL66 Prog 9)")
print("=" * 55)
print(f"\nDataset shape : {X.shape}")
print(f"Classes       : {list(class_names)}")
print(f"Features      : {list(iris.feature_names)}")

# ── 2. Train / Test Split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

print(f"\nTrain samples : {len(X_train)}")
print(f"Test  samples : {len(X_test)}")

# ── 3. Feature Scaling ────────────────────────────────────────────────────────
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── 4. Train KNN ──────────────────────────────────────────────────────────────
K = 5
knn = KNeighborsClassifier(n_neighbors=K, metric='euclidean')
knn.fit(X_train, y_train)
print(f"\nModel : KNeighborsClassifier(n_neighbors={K}, metric='euclidean')")

# ── 5. Predict ────────────────────────────────────────────────────────────────
y_pred = knn.predict(X_test)

# ── 6. Print Correct & Wrong Predictions ──────────────────────────────────────
print("\n" + "─" * 55)
print(f"{'#':<5}{'Actual':<15}{'Predicted':<15}{'Result'}")
print("─" * 55)
correct, wrong = 0, 0
for i, (actual, pred) in enumerate(zip(y_test, y_pred)):
    result = "Correct" if actual == pred else "WRONG"
    if actual == pred: correct += 1
    else: wrong += 1
    print(f"{i+1:<5}{class_names[actual]:<15}{class_names[pred]:<15}{result}")

# ── 7. Metrics ────────────────────────────────────────────────────────────────
acc = accuracy_score(y_test, y_pred)
cm  = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 55)
print(f"Accuracy  : {acc * 100:.2f}%")
print(f"Correct   : {correct} / {len(y_test)}")
print(f"Wrong     : {wrong} / {len(y_test)}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# ── 8. Accuracy vs K ──────────────────────────────────────────────────────────
k_values, k_accs = [], []
for k_val in range(1, 16, 2):
    m = KNeighborsClassifier(n_neighbors=k_val, metric='euclidean')
    m.fit(X_train, y_train)
    k_values.append(k_val)
    k_accs.append(accuracy_score(y_test, m.predict(X_test)) * 100)

# ── 9. Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("KNN Classifier — Iris Dataset (VTU 22CDL66 Program 9)",
             fontsize=13, fontweight='bold')

# Confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=class_names, yticklabels=class_names)
axes[0].set_title(f"Confusion Matrix (K={K})")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# Accuracy vs K
axes[1].plot(k_values, k_accs, marker='o', color='steelblue', linewidth=2, markersize=6)
axes[1].axvline(x=K, color='tomato', linestyle='--', label=f'K={K} (selected)')
axes[1].set_title("Accuracy vs K value")
axes[1].set_xlabel("K")
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_ylim(60, 105)
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("knn_results.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved as knn_results.png")
