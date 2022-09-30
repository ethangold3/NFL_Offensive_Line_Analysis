from urllib.request import urlopen
from bs4 import BeautifulSoup
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing



from openpyxl import Workbook
import re
import csv
import lxml

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold


def inte(string):
    return int(string)
def beg(string):
    if type(string) == str:
        idx = string.find('(')
        return string[:idx-1]

    else:
        return string

def pos(string):
    if string > 0:
        return 1
    if string < 0:
        return 0
    if string == 0:
        return 0

def func():
    df1 = pd.read_csv('NFL_CAP_DATA.csv')
    df = df1.copy()
    df['Wins'] = df['Wins'].apply(inte)
    df['Vegas Total'] = df['Vegas Total'].apply(beg)
    df['Vegas Total'] = df['Vegas Total'].apply(float)
    df['Y'] = df['Wins'] - df['Vegas Total']
    df['Y'] = df['Y'].apply(pos)
    return df

def model(df):
    q = df[['OL', 'total', 'QB', 'Vegas Total', 'Defense', 'RB']]
    scaler = preprocessing.StandardScaler().fit(q)
    x = scaler.transform(q)

   # x = x.drop(['Offense', 'Defense'], axis=1)
    y = df['Y']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.32, random_state=2)

    p2_features = PolynomialFeatures(degree=2)
    p2_train = p2_features.fit_transform(x_train)
    p2_test = p2_features.fit_transform(x_test)


    clf = LassoCV(cv=3, random_state=1).fit(p2_train, y_train)


    print(clf.score(p2_train, y_train))
    print(x)
    print(q)




def bottom(num):
    if num < .09:
        return 1
    else:
        return 0




if __name__ == '__main__':
    df = func()
    model(df)
    #plt.scatter(y, x)
    #plt.show()