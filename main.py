# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 06:12:31 2026

@author: zainali
"""

import pandas as pd

mh = pd.read_csv("Mental_Health_Dataset_Cleaned.csv")

print(mh.head())
print(mh.info())

mh_dt = mh.copy()

print("___Rows Removed")
mh_dt = mh_dt.drop(columns=["Gender", "Social_Media_Platform"])

print(mh_dt.head())
print(mh_dt.info())

print("__Separate Data and Label__")
X = mh_dt.drop(columns=["Happiness_Index_3highest"])
y = mh_dt["Happiness_Index_3highest"]

print("Training Data (X)")
print(X.head())

print("Label (y)")
print(y.head())


print("___ SKlearn trian test split ___")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

print(X_train.head())
print(X_test.head())
print(y_train.head())
print(y_test.head())

print("___Seaborn, Matplotlib")

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
sns.histplot(X_train["Age"], kde=True)
plt.title("Distribution of Age in the Mental Health Dataset")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

print("___Train Decision Tree___")

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

print("___Confusion Matrix___")
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)

print("__Plot Decision Tree__")
plt.figure(figsize=(7,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Mental Health Dataset Decision Tree Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(
    x=X_train["Daily_Screen_Time_hrs"],
    y=X_train["Sleep_Quality_1_10"]
)

plt.title("Daily Screen Time vs Sleep Quality")
plt.xlabel("Daily Screen Time (Hours)")
plt.ylabel("Sleep Quality (1–10)")
plt.show()

from sklearn.tree import plot_tree

plt.figure(figsize=(16,10))
plot_tree(
    tree,
    feature_names=X.columns,
    class_names=["1", "2", "3"],
    filled=True,
    rounded=True,
    fontsize=6
)