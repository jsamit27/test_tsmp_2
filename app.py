import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

import pandas as pd
import missingno as msno

#matplotlib.use('agg')
import matplotlib.pyplot as plt

import seaborn as sns
import os
import numpy as np
from fasteda import fast_eda
import plotly.express as px
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf
from scipy.stats import boxcox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import shapiro
from sklearn.metrics import mean_absolute_error, mean_squared_error





import streamlit as st

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('Trend Minder Stock Predictor (TMSP)')

st.subheader('By: Hemraj Yadav and Amit Sai Jitta')





#combined_data=pd.read_csv('comb.csv')

combined_data = pd.read_csv('reduced.csv')
combined_data = combined_data.iloc[:, 1:]

combined_data['date']=pd.to_datetime(combined_data['Date'])
combined_data['Date'] = pd.to_datetime(combined_data['Date'])
combined_data.set_index('Date', inplace=True)

input_ticker = st.text_input("Enter Stock ticker")


# Check if input_ticker is a valid column name in the DataFrame

    # Access the column from the DataFrame
close=combined_data[combined_data['Ticker'] == input_ticker]
#close = df[input_ticker]
st.write(f"Data for {input_ticker}:")
st.write(close)



#start_date = '2020-01-01'
#end_date = '2020-12-31'

fig=plt.figure( figsize = (12,6))
st.subheader('closing price')

plt.plot(close['Close'])

#filtered_data = close[(close['Date'] >= start_date) & (close['Date'] <= end_date)]

#plt.plot(filtered_data['Close']

plt.title(f"Time series for {input_ticker}")
plt.xlabel("Index")
plt.ylabel("Close Price")
st.pyplot() 

st.subheader('Volume')
#fig=plt.figure(figsize=(20,12))
plt.plot(close['Volume'])
plt.title(f"Time series for {input_ticker}")
plt.xlabel("Index")
plt.ylabel("Close Price")
st.pyplot() 



start_date= st.text_input(' Enter start date')
end_date= st.text_input('Enter end day')


filtered_data = close[(close['date'] >= start_date) & (close['date'] <= end_date)]


st.subheader('Certain Time frame Close values')

plt.plot(filtered_data['Close'])
plt.title(f"Time series for {input_ticker}")
plt.xlabel("Index")
plt.ylabel("Close Price")
st.pyplot() 


st.subheader('Certain Time frame Volume values')

plt.plot(filtered_data['Volume'])
plt.title(f"Time series for {input_ticker}")
plt.xlabel("Index")
plt.ylabel("Close Price")
st.pyplot() 


moving_avg_50 = close['Close'].rolling(window=50).mean()

# Calculate the 200-day moving average for each stock
moving_avg_200 = close['Close'].rolling(window=200).mean()

st.subheader('Moving average')


plt.plot(close['Close'],'g', label='Original price')
plt.plot(moving_avg_50, 'r', label='50-day Moving Average')

# Plot moving_avg_200 with blue color and label
plt.plot(moving_avg_200, 'b', label='200-day Moving Average')
plt.legend()

plt.title(f"Moving Average for {input_ticker}")
plt.xlabel("Index")
plt.ylabel("Close Price")
st.pyplot() 


# Decomposing the time series
if(input_ticker):

    st.subheader('Decomposition')

    result = seasonal_decompose(close['Close'], model='additive', period=365)

    #plt.figure(figsize=(20, 20))
    plt.plot(result.observed, label='Original', color='blue')
    plt.legend(loc='upper left')
    plt.subplot(412)
    plt.plot(result.trend, label='Trend', color='red')
    plt.legend(loc='upper left')
    plt.subplot(413)
    plt.plot(result.seasonal, label='Seasonal', color='green')
    plt.legend(loc='upper left')
    plt.subplot(414)
    plt.plot(result.resid, label='Residual', color='purple')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()
    st.pyplot()



    ts = close


    ts_removed = ts.drop(columns=['Ticker'])
    #st.write(ts)
    result = adfuller(ts_removed['Close'])

    #st.write(result)

        # Extract and print the test results

    st.subheader('ADF STATISTICS')

    adf_statistic = result[0]
    p_value = result[1]
    critical_values = result[4]


    st.write(f'ADF Statistic: {adf_statistic}')
    st.write(f'p-value: {p_value}')
    st.write('Critical Values:')
    for key, value in critical_values.items():
        st.write(f'   {key}: {value}')

        # Interpret the test results
    if p_value < 0.05:
        st.write('Result: Series is likely stationary (reject null hypothesis)')
    else:
        st.write('Result: Series is likely non-stationary (fail to reject null hypothesis)')


    #cHECKING TREND AND SEASONALITY

    ts = ts_removed['Close']

        # Calculate autocorrelation
    autocorr = acf(ts, fft=False)

        # Check for trend (autocorrelation in the first lag)
    trend_detected = autocorr[1] > 0.5  # Adjust threshold as needed

        # Check for seasonality (autocorrelation at seasonal lags)
    seasonal_detected = any(autocorr[lag] > 0.5 for lag in range(7, 31))  # Adjust lag range as needed

    st.subheader('Trend detection')

    if trend_detected:
        st.write('Trend detected')
    else:
        st.write('No trend detected')

    st.subheader('Seasonality detection')

    if seasonal_detected:
        st.write('Seasonality detected')
    else:
        st.write('No seasonality detected')
        

    #model
    from tensorflow.keras.models import load_model

    if input_ticker == "TSLA":
        model = load_model('tsla_model.h5')
    elif input_ticker == "AAPL":
        model = load_model('aapl_model.h5')
    elif input_ticker == "AMZN":
        model = load_model('amzn_model.h5')
    elif input_ticker == 'F':
        model = load_model('f_model.h5')





    # model

    data_training= pd.DataFrame(ts[0:int(len(ts)*0.70)])
    data_testing = pd.DataFrame(ts[int(len(ts)*0.70):int(len(ts))])


    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))

    data_training_array= scaler.fit_transform(data_training)

    x_train= []
    y_train= []

    for i in range(100, data_training_array.shape[0]):
        x_train.append(data_training_array[i-100:i])
        y_train.append(data_training_array[i,0])
        
    x_train, y_train= np.array(x_train),np.array(y_train)



    past_100_days= data_training.tail(100)

    final_df= past_100_days.append(data_testing,ignore_index= True )

    input_data= scaler.fit_transform(final_df)

    x_test=[]
    y_test=[]

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i,0])

    x_test, y_test= np.array(x_test),np.array(y_test)

    # predictions

    y_predicted= model.predict(x_test)

    scale_factor=1/scaler.scale_
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    st.title("Original vs Predicted ")

    plt.figure(figsize=(12,6))
    plt.plot(y_test,'b', label="original price")
    plt.plot(y_predicted,"r",label='predicted price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    st.pyplot()


    st.subheader("30-Day Forecasting ")

    df=pd.read_csv('comb.csv')

    df=df[df['Ticker'] == input_ticker]

    df1=df.reset_index()['Close']

    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler(feature_range=(0,1))
    df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

    ##splitting dataset into train and test split
    training_size=int(len(df1)*0.65)
    test_size=len(df1)-training_size
    train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

    import numpy
    # convert an array of values into a dataset matrix
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return numpy.array(dataX), numpy.array(dataY)


    # reshape into X=t,t+1,t+2,t+3 and Y=t+4
    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)

    # reshape input to be [samples, time steps, features] which is required for LSTM
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

    # model loading comes here 

    if input_ticker == "TSLA":
        n_model = load_model('test_TSLA_model.h5')
    elif input_ticker == "AAPL":
        n_model = load_model('test_AAPL_model.h5')
    elif input_ticker == "AMZN":
        n_model = load_model('test_AMZN_model.h5')
    elif input_ticker == 'F':
        n_model = load_model('test_f_model.h5')






    x_input=test_data[len(test_data)-100:].reshape(1,-1)
    x_input.shape

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()

    # demonstrate prediction for next 10 days
    from numpy import array

    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
        
        if(len(temp_input)>100):
            #print(temp_input)
            x_input=np.array(temp_input[1:])
            print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            #print(x_input)
            yhat = n_model.predict(x_input, verbose=0)
            print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1
        

    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)

    plt.plot(day_new,scaler.inverse_transform(df1[len(df1)-100:]))
    plt.plot(day_pred,scaler.inverse_transform(lst_output))

    st.pyplot()
        

    st.subheader('Whole graph with Forecasting')

    df3=df1.tolist()
    df3.extend(lst_output)

    df3=scaler.inverse_transform(df3).tolist()

    plt.plot(df3)

    st.pyplot()