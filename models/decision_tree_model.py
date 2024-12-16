# models/decision_tree_model.py

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt
import io
import base64


def run_decision_tree_model(df, features, target):
    # Prepare features and target
    X = df[features]
    y = df[target]

    # Initialize and train the Decision Tree classifier
    clf = DecisionTreeClassifier(criterion="gini", random_state=42)
    clf.fit(X, y)

    # Get the decision tree rules as text
    tree_rules = export_text(clf, feature_names=features)

    # Plot the Decision Tree
    plt.figure(figsize=(12, 8))
    plot_tree(clf, feature_names=features, class_names=["drugA", "drugB", "drugC", "drugX", "drugY"], filled=True)

    # Save the plot to a bytes buffer
    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()  # Close the plot to avoid display issues

    return tree_rules, plot_url
