import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn import tree
import graphviz
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# convert the csv to pandas dataframe
df = pd.read_csv("./data/customer_churn.csv")

#Data Preprocessing

#Separating the churn from the dataset
X = df.drop("Churn", axis=1)   # Features
y = df["Churn"]                # Target

#Splitting the training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Data visualization
plt.figure(figsize=(15,17))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("./images/heatmap.png")


# Modeling

# Initialize Random Forest
rf = RandomForestClassifier(
    n_estimators=100,   # Number of trees
    max_depth=None,     # Let trees expand until all leaves are pure
    random_state=42,
    class_weight="balanced"  # useful if dataset is imbalanced
)

# Fit model
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Get feature importances
importances = rf.feature_importances_
features = X.columns

# Sort by importance
indices = np.argsort(importances)[::-1]

# Plot
plt.figure(figsize=(10,18))
plt.title("Feature Importance")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), features[indices], rotation=90)
plt.savefig('./images/random_forester_model.png')

# Get one tree from the forest (e.g., the first one)
estimator = rf.estimators_[0]


# plot the tree
plt.figure(figsize=(30,10))
tree.plot_tree(
    estimator,
    feature_names=X.columns,
    class_names=["No Churn", "Churn"],
    filled=True
)
plt.savefig("./images/decision-tree.png")
