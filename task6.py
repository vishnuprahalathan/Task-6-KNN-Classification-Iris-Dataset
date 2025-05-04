
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from matplotlib.colors import ListedColormap

csv_path = "C:\\Users\\Vishnu Prahalathan\\Desktop\\Iris.csv"  
df = pd.read_csv(csv_path)


print("Dataset Sample:")
print(df.head())

print("\nDataset Statistics:")
print(df.describe())

print("\nMissing Values:", df.isnull().sum().sum())


X = df.drop('Species', axis=1).values
y = df['Species'].values

le = LabelEncoder()
y = le.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

k_range = range(1, 21)
cv_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    cv_scores.append(scores.mean())


plt.figure(figsize=(10, 6))
plt.plot(k_range, cv_scores, marker='o')
plt.title("Cross-Validation Accuracy vs. K")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.xticks(k_range)
plt.grid(True)
plt.show()


best_k = k_range[np.argmax(cv_scores)]
print(f"\n🏆 Best K: {best_k} (CV Accuracy: {max(cv_scores)*100:.2f}%)")


model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


acc = accuracy_score(y_test, y_pred)
print(f"\n🎯 Test Accuracy: {acc * 100:.2f}%")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_test, y_pred)
print("\n📊 Confusion Matrix:")
print(cm)


cv_full = cross_val_score(model, X, y, cv=5)
print(f"\n🔁 Cross-Validation Scores: {cv_full}")
print(f"📈 Mean CV Accuracy: {cv_full.mean() * 100:.2f}% ± {cv_full.std() * 100:.2f}%")


X_vis = X[:, :2]
X_vis = StandardScaler().fit_transform(X_vis)

X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(
    X_vis, y, test_size=0.2, random_state=42, stratify=y
)


vis_model = KNeighborsClassifier(n_neighbors=best_k)
vis_model.fit(X_train_vis, y_train_vis)


h = 0.02
x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))


Z = vis_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.5)
sns.scatterplot(x=X_vis[:, 0], y=X_vis[:, 1], hue=y, palette=cmap_bold, edgecolor='k', s=80)
plt.title(f"Decision Boundary with K={best_k}")
plt.xlabel("Sepal Length (Standardized)")
plt.ylabel("Sepal Width (Standardized)")
plt.grid(True)
plt.legend(title="Species")
plt.tight_layout()
plt.show()