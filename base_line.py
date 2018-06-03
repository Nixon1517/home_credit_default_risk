import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def encode(data):
    label_encoder = LabelEncoder()
    to_encode = ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY']
    for col in to_encode:
        data[col] = label_encoder.fit_transform(data[col])
    return data


def logistic_regression(data, validation_percent=0.1):
    ans = data['TARGET']
    data = data.drop(columns='TARGET')
    x_train, x_test, y_train, y_test = train_test_split(data, ans, test_size=validation_percent)
    log_reg = LogisticRegression()
    log_reg.fit(x_train, y_train)
    predictions = log_reg.predict(x_test)
    score = log_reg.score(x_test, y_test)
    return score


def k_neighbors(data, validation_percent=0.1, n_neightbors=5):
    ans = data['TARGET']
    data = data.drop(columns='TARGET')
    x_train, x_test, y_train, y_test = train_test_split(data, ans, test_size=validation_percent)
    neigh = KNeighborsClassifier(n_neighbors=n_neightbors)
    neigh.fit(x_train, y_train)
    predictions = neigh.predict(x_test)
    score = neigh.score(x_test, y_test)
    return score



train_df = pd.read_csv( "C://Users//chris//.kaggle//competitions//home-credit-default-risk//application_train.csv")
small_data = train_df[['TARGET', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']].copy()

small_data = encode(small_data)


print("Logistic regression score: ", logistic_regression(small_data))
print("KNN Score: ", k_neighbors(small_data))
print("done testing")



