# Installation Instructions
### Install the following packages: 

- pip install streamlit
- pip install yfinance    
- pip install datetime
- pip install pandas
- pip install numpy
- pip install sklearn
- pip install keras
- pip install matplotlib
- pip install beautifulsoup4
- pip install ssl
- pip install requests


# Run Code
To run code, navigate to folder and enter "streamlit run main.py" in terminal.
Follow the Local Url to open the web app

# Code Structure
I used streamlit, an open source web app api to help me easily display my script in an interactive app.
Users will input a valid stock ticker listed in the S&P500.
I webscraped the description of each stock off of yahoo finance and got raw data from yfinance api.
A graph of open and close price of everyday is made.
Next, I have implemented a machine learning model that will make stock price predictions based on the closing price within a time frame.
I output this graph and compare it with the actual closing prices.
I keep track of how long the model runs using time package.
Finally, I print out the predicted price of the next day's close.




