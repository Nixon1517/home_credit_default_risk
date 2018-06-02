import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split




def logistic_regression(data, validation_percent = 0.1):
    ans = data['TARGET']
    data = data.drop(columns='TARGET')
    x_train, x_test, y_train, y_test = train_test_split(data, ans, test_size=validation_percent)
    log_reg = LogisticRegression()
    log_reg.fit(x_train, y_train)
    predictions = log_reg.predict(y_train)
    score = log_reg.score(y_test, y_train)
    return score


train_df = pd.read_csv( "C://Users//chris//.kaggle//competitions//home-credit-default-risk//application_train.csv")
small_data = train_df[['TARGET', 'FLAG_OWN_CAR']].copy()

print(small_data.head())

score = logistic_regression(small_data)
print(score)



