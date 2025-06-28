# Stock Price Prediction Showcase

This project demonstrates a Python-based application for stock price prediction using LSTM (Long Short-Term Memory) neural networks. It serves as a showcase of Python programming skills, including data processing, machine learning, and GUI development.

## Features
![Application Screenshot](screenshot.png)

- **Stock Data Retrieval**: Automatically downloads historical stock data using the `yfinance` library.
- **LSTM Model Training**: Trains an LSTM model to predict stock prices based on historical data.
- **Future Price Prediction**: Predicts future stock prices for up to 30 days.
- **Interactive GUI**: A user-friendly interface built with `customtkinter` for selecting stocks, viewing historical data, and predictions.
- **Dynamic Visualization**: Real-time plotting of stock prices and predictions using `matplotlib`.

## Project Structure

- **`APP.PY`**: The main application file that integrates the GUI and visualization components.
- **`download_stock_data.py`**: Contains functions for downloading stock data, training the LSTM model, and making predictions.

## How to Run

1. **Install Dependencies**:
   Ensure you have Python installed. Install the required libraries using:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Create a `requirements.txt` file with the necessary dependencies such as `yfinance`, `tensorflow`, `numpy`, `matplotlib`, `customtkinter`, etc.)*

2. **Run the Application**:
   Execute the main application file:
   ```bash
   python APP.PY
   ```

3. **Interact with the GUI**:
   - Select a stock ticker from the sidebar.
   - Choose a time range for visualization.
   - View historical data, LSTM predictions, and future price forecasts.

## Key Components

### `APP.PY`

- **GUI**: Built with `customtkinter`, it provides an intuitive interface for users to interact with the application.
- **Visualization**: Uses `matplotlib` to plot stock prices, historical predictions, and future forecasts.
- **Ticker Management**: Allows users to filter and select stock tickers dynamically.

### `download_stock_data.py`

- **Data Retrieval**: Fetches historical stock data using `yfinance`.
- **LSTM Model**: Implements an LSTM neural network for time-series prediction.
- **Future Predictions**: Generates future stock price predictions based on the trained model.

## Future Improvements

- **Enhanced Model**: Experiment with additional features like sentiment analysis or technical indicators to improve prediction accuracy.
- **Cloud Integration**: Deploy the application on a cloud platform for broader accessibility.
- **Mobile Support**: Develop a mobile-friendly version of the application.
- **Backtesting**: Add functionality to evaluate the model's performance on unseen data.

This project is a comprehensive showcase of Python programming skills, combining machine learning, data visualization, and GUI development into a cohesive application.