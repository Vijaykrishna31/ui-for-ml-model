import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_text, plot_tree
import matplotlib.pyplot as plt
import io
import base64
import seaborn as sns
from sklearn.metrics import confusion_matrix


def run_random_forest_model(df, features, target):
    # Split data into features and target
    X = df[features]
    y = df[target]

    # Initialize and train the Random Forest classifier
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X, y)

    # Export rules for one tree in the Random Forest
    tree_rules = export_text(rf_clf.estimators_[0], feature_names=features)

    # Create image buffers
    tree_images = []
    for i in range(3):  # Generate images for the first 3 trees
        fig, ax = plt.subplots(figsize=(5, 4))
        plot_tree(
            rf_clf.estimators_[i],
            feature_names=features,
            class_names=["No", "Yes"],
            filled=True,
            ax=ax,
        )
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        tree_images.append(base64.b64encode(buf.getvalue()).decode())
        plt.close(fig)  # Close the plot to free memory

    # Generate confusion matrix and plot it
    y_pred = rf_clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    cm_plot_url = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)  # Close the plot to free memory

    return tree_rules, tree_images, cm, cm_plot_url
