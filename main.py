import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
# Set the page title
st.set_page_config(layout="wide")
st.title('Current Stock Prices with Prediction Model')

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
period = st.selectbox('Select period of data:', ['1d', '1wk', '1mo', '3mo', '6mo', '1y', '5y', 'max'])
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

            TODAY = date.today().strftime("%Y-%m-%d")

            st.title('Stock Forecast App')

            selected_stock = ticker

            start_year = st.slider('Select start year for prediction:', 2000, date.today().year, 2000)

            # Konwersja na datę z wybranym rokiem (zachowując 01-01)
            START = f"{start_year}-01-01"
            n_years = st.slider('Years of prediction:', 1, 4)
            period = n_years * 365


            @st.cache_data
            def load_data(ticker, START):
                data = yf.download(ticker, START, TODAY)
                data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]  ##
                data.reset_index(inplace=True)
                return data


            data_load_state = st.text('Loading data...')
            data = load_data(selected_stock, START)
            data_load_state.text('Loading data... done!')
            st.subheader('Raw data')
            st.write(data.tail())


            # Plot raw data
            def plot_raw_data():
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
                fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
                fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
                st.plotly_chart(fig)


            plot_raw_data()
            st.write(f"Columns: {data.columns}")
            # Predict forecast with Prophet.
            df_train = data[['Date', 'Close']]
            df_train.columns = ['ds', 'y']

            m = Prophet()
            m.fit(df_train)
            future = m.make_future_dataframe(periods=period)
            forecast = m.predict(future)

            # Show and plot forecast
            st.subheader('Forecast data')
            st.write(forecast.tail())

            st.write(f'Forecast plot for {n_years} years')
            fig1 = plot_plotly(m, forecast)
            actual_color = '#FFFFFF'
            fig1.update_traces(
                selector=dict(mode='markers'),
                marker=dict(color=actual_color, size=2)
            )
            st.plotly_chart(fig1)

            # st.write("Forecast components")
            # fig2 = m.plot_components(forecast)
            # st.write(fig2)

            fig2 = m.plot_components(forecast)
            fig2.patch.set_facecolor('black')
            for ax in fig2.axes:
                ax.set_facecolor('black')
                ax.tick_params(axis='both', colors='white')
                ax.set_xlabel(ax.get_xlabel(), color='white')
                ax.set_ylabel(ax.get_ylabel(), color='white')

            st.write("Forecast components")
            st.pyplot(fig2)
















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
