import pandas as pd
import math
import random
import os
import numpy as np
from sklearn import preprocessing, svm
# from sklearn import cross_validation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt


def getStockData(stock="zggw"):
    stockfile = "zggw2017.csv"
    allData = pd.read_csv(stockfile)
    print("read original DF shape ==>", allData.shape)
    dataLength = 261
    #  allDataLength = len(allData)
    #  firstDataElem = math.floor(random.random() * (allDataLength - dataLength))
    firstDataElem = 250    # 511 =  firstDataElem + dataLength  accuracy = 0.64

    mlData = allData[0:firstDataElem + dataLength]

    def FormatForModel(dataArray):
        dataArray = dataArray[['Open', 'High', 'Low', 'Volume', 'Adj Close']]
        dataArray['HL_PCT'] = (dataArray['High'] - dataArray['Adj Close']
                               ) / dataArray['Adj Close'] * 100.0
        dataArray['PCT_change'] = (dataArray['Adj Close'] - dataArray['Open']
                                   ) / dataArray['Open'] * 100.0
        dataArray = dataArray[['Adj Close', 'HL_PCT', 'PCT_change', 'Volume']]
        dataArray.fillna(-99999, inplace=True)
        return dataArray

    mlData = FormatForModel(mlData)

    forecast_col = 'Adj Close'
    print(f">>> mlData Random!! Length is ===> {firstDataElem+dataLength}")
    forecast_out = int(math.ceil(0.12 * dataLength))
    #  forecast_out = 15
    print(f">>> forecast_out value ===> {forecast_out} ")

    mlData['label'] = mlData[forecast_col].shift(-forecast_out)
    print(" >>>probing the mlData last 30 ==> ", mlData[-30:])
    #  print(" >>>probing the label last 15 ==> ", mlData['label'][-15:])
    mlData.dropna(inplace=True)
    #  print(mlData.describe())

    print(f">>> mlData df last 5 rows ==> {mlData[-5:]}\n")
    X = np.array(mlData.drop(['label'], 1))
    print(f">>> X array last 5 after drop label ==> {X[-5:]}\n")
    X = preprocessing.scale(X)
    print(f">>> X last 5 elements After preprocessing scale ==> {X[-5:]}\n")
    X_data = X[-dataLength:]
    print(f">>> X_data head 5  ==> {X_data[:5]}\n")
    print(f">>> X_data last 5  ==> {X_data[-5:]}\n")
    X = X[:-dataLength]
    data = mlData[-dataLength:]
    mlData = mlData[:-dataLength]
    y = np.array(mlData['label'])
    print(f">>> y from mlData label last 5 elements ==> {y[-5:]}\n")

    X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=0.3)

    #  Only employing the LinearRegression , should we try the Bayers Naiive
    # and SVM , and most of all ,got the volume column be included, and the
    # multi-dimension
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(f">>> accuracy measured on test set ==> {accuracy}")
    # have all the sequence of pridictions along the X_data
    # dump into a prediction array, finally  jsonfy and transfering
    prediction = clf.predict(X_data)
    data = data[['Adj Close']]
    # how about the below, gotcha rename is wut the shit
    #  data = data.rename(columns={'Adj Close': 'EOD'})
    data['prediction'] = prediction[:]
    return data


def plot_graph(dat_fm):
    """plot dataFrame columns """
    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.set_xlabel("Daily Series")
    axes.set_ylabel("Prices")
    axes.set_title("EOD and Predictions")
    eod_list = dat_fm['EOD'].values * 10.0
    _len = len(eod_list)
    x = np.linspace(0, _len, _len)
    pl1, = plt.plot(x, eod_list, 'g--')
    pre_list = dat_fm['prediction'].values * 10.0
    pl2, = plt.plot(x, pre_list, 'b-')
    axes.legend(handles=[pl1, pl2], labels=['Close', 'Predicts'], loc='best')
    # plt.legend()
    plt.show()


if __name__ == '__main__':
    zggw_df = getStockData()
    print(zggw_df.shape)
    print(zggw_df.iloc[::, 0][:5])
    zggw_df.to_csv("zggw_predict.csv")
    #  print(msft_df[-5:])
    # plot_graph(msft_df)
