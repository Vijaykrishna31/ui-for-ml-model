import pandas as pd
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

def run_multilinear_regression_model(df, features, target):
    # Extract the selected features and target
    X = df[features]
    y = df[target]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # Initialize and train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_predict_train = model.predict(X_train)
    y_predict_test = model.predict(X_test)

    # Generate 3D plot if there are exactly two features
    plot_url = None
    if len(features) == 2:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Plot training data and predictions
        ax.scatter(X_train[features[1]], X_train[features[0]], y_train, color="green", label="Training data")
        ax.scatter(X_train[features[1]], X_train[features[0]], y_predict_train, color="blue", label="Predictions")

        # Labels and legend
        ax.set_title("3D Plot")
        ax.set_xlabel(features[1])
        ax.set_ylabel(features[0])
        ax.set_zlabel(target)
        plt.legend()

        # Save plot to a bytes buffer and encode as base64
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

    # Return the model summary and plot URL
    model_summary = {
        "coefficients": dict(zip(features, model.coef_)),
        "intercept": model.intercept_
    }
    return model_summary, plot_url
