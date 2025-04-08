import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression

# Set the page title
st.set_page_config(layout="wide")
st.title('📊 Current Stock Prices with Prediction Model')

# Function to fetch stock data
def get_stock_data(ticker, period):
    stock = yf.Ticker(ticker)
    stock_data = stock.history(period=period)  # Fetch data based on the period
    return stock_data

# Function to predict stock price based on linear regression
def predict_stock_price(data):
    # Prepare the data
    data['Date'] = data.index
    data['Date'] = data['Date'].map(datetime.toordinal)  # Convert date to integer (ordinal)

    # Select variables
    X = np.array(data['Date']).reshape(-1, 1)  # Date as independent variable
    y = np.array(data['Close'])  # Closing price as dependent variable

    # Create a linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict the stock price for the next day
    next_day = np.array([datetime.now().toordinal() + 1]).reshape(-1, 1)
    predicted_price = model.predict(next_day)[0]

    return predicted_price

# Options for selecting the period (only for the selected ticker)
period = st.selectbox('Select period:', ['1d', '1wk', '1mo', '3mo', '6mo', '1y', '5y', 'max'])

# List of popular stocks
popular_stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NFLX']

# Sidebar with popular stocks
st.sidebar.header('Predictions for Popular Stocks')

for stock in popular_stocks:
    try:
        # Fetch data for popular stocks (always 1d)
        stock_data = get_stock_data(stock, '1d')  # Set the period to 1d

        # Display the last closing price
        last_price = stock_data['Close'].iloc[-1]
        open_price = stock_data['Open'].iloc[0]
        change = last_price - open_price
        pct_change = (change / open_price) * 100

        # Predict the price for the next day
        predicted_price = predict_stock_price(stock_data)

        # Determine the color of the change (red for a decrease, green for an increase)
        color = 'green' if pct_change > 0 else 'red'

        # Determine the arrow (up for an increase, down for a decrease) in Unicode format
        arrow = '↑' if pct_change > 0 else '↓'

        # Display results for popular stocks
        st.sidebar.subheader(f"{stock} - {last_price:.2f} USD")
        st.sidebar.markdown(f"Change: <span style='color:{color};'>{change:.2f} USD ({pct_change:.2f}%) {arrow}</span>", unsafe_allow_html=True)

    except Exception as e:
        st.sidebar.error(f"Error fetching data for {stock}: {e}")

# Function to fetch data for multiple stocks
def get_multiple_stock_data(tickers, period):
    stock_data = {}
    for ticker in tickers:
        stock_data[ticker] = get_stock_data(ticker, period)
    return stock_data

# Create tabs
option = st.selectbox('Select option:', ['Single Stock Overview', 'Multiple Stock Comparison'])

if option == 'Single Stock Overview':
    # Input for the ticker symbol
    ticker = st.text_input('Enter stock ticker (e.g. AAPL, GOOGL, MSFT):', 'AAPL').upper()

    if ticker:
        try:
            # Fetch stock data (period based on user selection)
            stock_data = get_stock_data(ticker, period)

            # Display the last closing price
            last_price = stock_data['Close'].iloc[-1]
            open_price = stock_data['Open'].iloc[0]
            change = last_price - open_price
            pct_change = (change / open_price) * 100

            # Display predictions for the selected ticker in the main window
            st.subheader(f"{ticker} - {last_price:.2f} USD")
            color = 'green' if pct_change > 0 else 'red'
            arrow = '↑' if pct_change > 0 else '↓'
            st.markdown(f"Change: <span style='color:{color};'>{change:.2f} USD ({pct_change:.2f}%) {arrow}</span>", unsafe_allow_html=True)

            # Predict the price for the next day
            predicted_price = predict_stock_price(stock_data)
            st.markdown(f"Prediction for the next day: {predicted_price:.2f} USD")

            # Add space between charts
            st.markdown("<br>", unsafe_allow_html=True)

            # Display the chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close'))
            fig.update_layout(title=f'{ticker} Chart for Period: {period}',
                              xaxis_title='Date',
                              yaxis_title='Price (USD)',
                              height=600)
            st.plotly_chart(fig)

            # Display the historical data table
            st.subheader('Historical Data')
            st.dataframe(stock_data[['Open', 'High', 'Low', 'Close', 'Volume']])

        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")

    else:
        st.info('Please enter a stock ticker.')

elif option == 'Multiple Stock Comparison':
    # Select stocks for comparison
    tickers = st.multiselect('Select stocks for comparison:', popular_stocks, default=['AAPL', 'GOOGL'])

    if tickers:
        try:
            # Fetch data for the selected stocks
            stock_data = get_multiple_stock_data(tickers, period)

            # Display the comparison chart
            fig = go.Figure()

            for ticker in tickers:
                # Add data for each stock
                fig.add_trace(go.Scatter(x=stock_data[ticker].index,
                                         y=stock_data[ticker]['Close'],
                                         mode='lines',
                                         name=ticker))

            fig.update_layout(title=f'Stock Comparison: {", ".join(tickers)}',
                              xaxis_title='Date',
                              yaxis_title='Price (USD)',
                              height=600)

            st.plotly_chart(fig)

            # Display the historical data tables
            st.subheader('Historical Data')
            for ticker in tickers:
                st.markdown(f"### {ticker}")
                st.dataframe(stock_data[ticker][['Open', 'High', 'Low', 'Close', 'Volume']])

        except Exception as e:
            st.error(f"Error fetching data for the selected stocks: {e}")

    else:
        st.info('Please select stocks for comparison.')
