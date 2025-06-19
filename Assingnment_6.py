# Step 1
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, multilabel_confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Step 2
data = sns.load_dataset('Iris')
print(data)

# Step 3: Split features and labels
X = data.drop('species', axis=1)  # Features
y = data['species']

# Label encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Step 5: Train Naïve Bayes Model
model = GaussianNB()
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Step 8: Accuracy, Precision, Recall, etc.
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
error_rate = 1 - acc

print("Accuracy:", acc)
print("Error Rate:", error_rate)
print("Precision:", precision)
print("Recall:", recall)

# Step 9: TP, FP, FN, TN for each class
print("\nTP, FP, FN, TN for each class:")
mcm = multilabel_confusion_matrix(y_test, y_pred)

for i, label in enumerate(le.classes_):
    tn, fp, fn, tp = mcm[i].ravel()
    print(f"\nClass: {label}")
    print(f"TP = {tp}")
    print(f"FP = {fp}")
    print(f"FN = {fn}")
    print(f"TN = {tn}")