# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 06:31:39 2026

@author: zainali
"""

import pandas as pd

news = pd.read_csv("News_Record_Labeled_Data_File.csv")

print(news.head())
print(news.info())
print(news["LABEL"].value_counts())

news_dt = news.copy()

print(news_dt.dtypes)

feature_cols = [col for col in news_dt.columns if col != "LABEL"]
news_dt[feature_cols] = news_dt[feature_cols].apply(pd.to_numeric, errors="coerce")
news_dt[feature_cols] = news_dt[feature_cols].fillna(0)

X_news = news_dt.drop(columns=["LABEL"])
y_news = news_dt["LABEL"]

print("Predictor Data:")
print(X_news.head())

print("Label Data:")
print(y_news.head())

from sklearn.model_selection import train_test_split

X_train_news, X_test_news, y_train_news, y_test_news = train_test_split(
    X_news,
    y_news,
    test_size=0.3,
    random_state=42,
    stratify=y_news
)

print("X_train shape:", X_train_news.shape)
print("X_test shape:", X_test_news.shape)
print("y_train shape:", y_train_news.shape)
print("y_test shape:", y_test_news.shape)

print(X_train_news.head())
print(X_test_news.head())
print(y_train_news.head())
print(y_test_news.head())

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,5))
sns.histplot(X_train_news["president"], kde=True)
plt.title("Distribution of the 'president' Feature in the Training Data")
plt.xlabel("Count of 'president'")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(
    x=X_train_news["company"],
    y=X_train_news["google"]
)
plt.title("Company vs Google Word Counts in Training Data")
plt.xlabel("Count of 'company'")
plt.ylabel("Count of 'google'")
plt.show()

from sklearn.tree import DecisionTreeClassifier

tree_news = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

tree_news.fit(X_train_news, y_train_news)

y_pred_news = tree_news.predict(X_test_news)

from sklearn.metrics import accuracy_score

news_accuracy = accuracy_score(y_test_news, y_pred_news)
print("NewsAPI Model Accuracy:", news_accuracy)


from sklearn.metrics import confusion_matrix

cm_news = confusion_matrix(y_test_news, y_pred_news, labels=tree_news.classes_)
print(cm_news)
print(tree_news.classes_)

plt.figure(figsize=(8,6))
sns.heatmap(
    cm_news,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=tree_news.classes_,
    yticklabels=tree_news.classes_
)
plt.title("NewsAPI Decision Tree Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()

from sklearn.tree import plot_tree

plt.figure(figsize=(22,12))
plot_tree(
    tree_news,
    feature_names=X_news.columns,
    class_names=tree_news.classes_,
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title("Decision Tree Visualization for NewsAPI Dataset")
plt.show()

importance_df = pd.DataFrame({
    "Feature": X_train_news.columns,
    "Importance": tree_news.feature_importances_
})

importance_df = importance_df.sort_values(by="Importance", ascending=False)
print(importance_df.head(10))