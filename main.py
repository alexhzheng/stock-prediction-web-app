import streamlit as st 
from datetime import date 
import yfinance as yf 
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import matplotlib.pyplot as plt
import time
from bs4 import BeautifulSoup
import urllib.request as urllib2
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# constants
START = "2015-01-01"
END = "2021-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# class with dunder methods __init__, __str__, and __set__
class Stock:
    def __init__(self, ticker, prediction):
        self.ticker = ticker

        # gets stock data from yfinace api
        self.data = yf.download(ticker, START, END)
        self.data.reset_index(inplace=True)

        # webscrape the name and description
        url = "https://finance.yahoo.com/quote/"+selected_stock+"/profile?p="+selected_stock
        page = urllib2.urlopen(url)
        soup = BeautifulSoup(page, 'html.parser')
        self.name = soup.find('h1', {'class':'D(ib) Fz(18px)'}).text.strip()
        self.description = soup.find('p',{'class':'Mt(15px) Lh(1.6)'}).text.strip()

        # will update the value after model is run
        self.prediction = prediction
        
    def __str__(self):
        return self.description

    def __set__(self, prediction, value):
        self.prediction = value



# returns a list of stock symbols to help me check if stock is valid and in S&P500
# only run once because cached
@st.cache
def load_stocks():
    df = pd.read_csv("snp500.csv")
    symbols = df['Symbol'].values.tolist()
    return symbols

# uses pyplot to plot the open and closing price of stock
def plot_raw_data():
    with st.beta_container():
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(stock.data['Date'], stock.data['Close'], color = "blue", label="stock_close")
        ax.set_title('Actual Price Graph')
        ax.set_xlabel('Time')
        ax.set_ylabel('Stock Price')
        ax.legend()
        st.pyplot(fig)

# return the predicted price and actual price of stock
def get_predicted_prices(ticker):
    # scale down all values so fit between 0 and 1
    scaler = MinMaxScaler(feature_range=(0,1))
    # only worry about close price
    scaled_data = scaler.fit_transform(stock.data['Close'].values.reshape(-1,1))

    # how many days want to look into past to predict next day/how many days to base prediction on
    prediction_days = 60

    x_train = []
    y_train = []

    # add scaled value into x_train -> add 60 values and 61st value so model can predict to learn what next value will be
    # add scaled value into y_train -> 61st data
    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x-prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    # put into np.array
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # reshape x_train to add one additional dimension
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    #build the model
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1)) # prediction of the next closing value

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    # test the model accuracy on existing data
    test_data = yf.download(ticker, END, TODAY)
    actual_prices = test_data['Close'].values 
    total_dataset = pd.concat((stock.data['Close'], test_data['Close']), axis = 0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1,1)
    model_inputs = scaler.transform(model_inputs)

    #make predictions on test data
    x_test = []
    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x-prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_prices = model.predict(x_test)

    predicted_prices = scaler.inverse_transform(predicted_prices)
    #predict next day
    real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs+1), 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    return (predicted_prices, actual_prices, prediction)

# uses pyploy to plot the predicted close price and actual close price of stock
def plot_predicted_data():
    with st.beta_container():
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(predicted_prices, color = "red", label="predicted_close")
        ax.plot(actual_prices, color = "blue", label="stock_close")
        ax.set_title('Prediction Price Graph')
        ax.set_xlabel('Time')
        ax.set_ylabel('Stock Price')
        ax.legend()
        st.pyplot(fig)


# code for streamlit web app
st.title("Stock Prediction App")
stocks = load_stocks()
selected_stock = st.text_input("Enter S&P500 ticker").upper()

if selected_stock != "" and selected_stock in stocks:
    
    stock = Stock(selected_stock, 0)
    st.subheader("Company Name")
    st.markdown(stock.name)
    st.subheader("Description")
    stock = Stock(selected_stock, 0)
    st.markdown(stock)

    #raw data table for most recent five days
    st.subheader('Raw data for ' + selected_stock)
    st.write(stock.data.tail())
    plot_raw_data()

    t0 = time.perf_counter()
    data_load_state2 = st.text("Calculating predicted prices for " + selected_stock + "...")
    mlmodel = get_predicted_prices(selected_stock)
    predicted_prices = mlmodel[0]
    actual_prices = mlmodel[1]
    stock.prediction = mlmodel[2][0][0]
    data_load_state2.text("Done!")
    st.subheader("Prediction for " + selected_stock)
    plot_predicted_data()
    t1 = time.perf_counter()
    time_state = st.text('Time taken for ML: ' + str(t1-t0) + " seconds")
    st.subheader("Prediction for next price: " + str(stock.prediction))
        




