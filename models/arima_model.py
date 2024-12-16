# models/arima_model.py

import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from statsmodels.tsa.arima.model import ARIMA

def run_arima_model(df, feature, target, forecast_steps=6):
    # Prepare the data
    df[feature] = pd.to_datetime(df[feature])
    df.set_index(feature, inplace=True)

    # Fit the ARIMA model
    model = ARIMA(df[target], order=(1, 1, 1))  # Adjust order as needed
    model_fit = model.fit()

    # Forecast
    forecast = model_fit.forecast(steps=forecast_steps)
    forecast_dates = pd.date_range(start=df.index[-1], periods=forecast_steps + 1, freq="M")[1:]

    # Plot the forecast
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df[target], label="Historical")
    plt.plot(forecast_dates, forecast, label="Forecast", linestyle="--")
    plt.title(f"{target} Forecast")
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.legend()

    # Save the plot to a bytes buffer
    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()  # Close the plot to avoid display issues

    return forecast.values, plot_url
