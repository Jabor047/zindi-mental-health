import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import Imputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV


df = pd.read_csv('Clean_test.csv')

model = joblib.load('gridcv.joblib')

X = df.drop(['depressed'], axis=1)

prediction = model.predict(X)

df2 = pd.read_csv('test.csv')

d = {'surveyid': df2['surveyid'], 'depressed': prediction}

df_predictions = pd.DataFrame(data=d)

df_predictions.to_csv('submit.csv', index=False)





