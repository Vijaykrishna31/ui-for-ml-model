# app.py

from flask import Flask, request, render_template, redirect, url_for, flash, session
import pandas as pd
import io
from models.arima_model import run_arima_model
from models.decision_tree_model import run_decision_tree_model
from models.random_forest_model import run_random_forest_model
from models.multilinear_regression_model import run_multilinear_regression_model

app = Flask(__name__)
app.secret_key = "idhellam oru polapa"  # Needed for flash messages


@app.route("/")
def upload_form():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        flash("No file part")
        return redirect(request.url)

    file = request.files["file"]

    if file.filename == "":
        flash("No selected file")
        return redirect(request.url)

    if file:
        data = pd.read_csv(io.StringIO(file.stream.read().decode("UTF-8")))

        # Store the DataFrame in the session as a CSV string
        session["data"] = data.to_csv(index=False)

        # Convert DataFrame to HTML for display
        data_html = data.to_html(classes="table table-striped")
        columns = data.columns.tolist()

        return render_template(
            "select_features.html", data_html=data_html, columns=columns
        )


@app.route("/run_model", methods=["POST"])
def run_model():
    model_type = request.form.get("model_type")
    features = request.form.getlist("feature")  # Get selected features as a list
    target = request.form.get("target")

    # Retrieve the DataFrame from the session
    df = pd.read_csv(io.StringIO(session["data"]))

    if model_type == "ARIMA":
        # ARIMA model (uses only one feature for the date)
        forecast, plot_url = run_arima_model(df, features[0], target)
        return render_template(
            "result_arima.html", plot_url=plot_url, forecast=forecast
        )

    elif model_type == "DecisionTree":
        # Run Decision Tree model with selected features
        tree_rules, plot_url = run_decision_tree_model(df, features, target)
        return render_template(
            "result_tree.html", plot_url=plot_url, tree_rules=tree_rules
        )

    elif model_type == "RandomForest":
        # Run Random Forest model with selected features
        tree_rules, tree_images, cm, cm_plot_url = run_random_forest_model(df, features, target)
        return render_template(
            "result_random_forest.html",
            tree_rules=tree_rules,
            tree_images=tree_images,
            cm=cm,
            cm_plot_url=cm_plot_url,
        )
        
    elif model_type == "MultilinearRegression":
        model_summary, plot_url = run_multilinear_regression_model(df, features, target)
        return render_template("result_multilinear.html", model_summary=model_summary, plot_url=plot_url)


    else:
        flash("Invalid model type selected.")
        return redirect(url_for("upload_form"))


if __name__ == "__main__":
    app.run(debug=True)
