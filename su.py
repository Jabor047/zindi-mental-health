import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import Imputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb


df = pd.read_csv('train.csv')

cols_drop = ['surveyid', 'survey_date']
# cols_drop = ['ent_employees','surveyid','survey_date','amount_given_mpesa','amount_received_mpesa',
# 			 'med_vacc_newborns','med_child_check','hh_totalmembers','early_survey','day_of_week','asset_niceroof',
# 			 'wage_expenditures', 'given_mpesa']

df = df.drop(cols_drop, axis=1)

# df = df.dropna(axis=1, thresh=len(df)*0.6)

# df = df.loc[:, df.isnull().mean()<0.6]
cols_em = df.columns[df.isnull().any()].tolist()


imputer = Imputer(missing_values=np.nan, strategy='mean', axis=1)

df[cols_em] = imputer.fit_transform(df[cols_em])

empty = pd.isnull(df).sum()

df.to_csv('Clean_train2.csv', index=False)

Y = df['depressed']
X=df.drop(['depressed'], axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.33, random_state=42)


model = xgb.XGBClassifier() #SVC(gamma='auto')
model.fit(X_train,Y_train)

predictions = model.predict(X_test)

df2 = pd.read_csv('Clean_test.csv')

print(accuracy_score(Y_test,predictions))

X2 = df2.drop('depressed', axis=1)

prediction = model.predict(X2)

df3 = pd.read_csv('test.csv')

d = {'surveyid': df3['surveyid'], 'depressed': prediction}

df_predictions = pd.DataFrame(data=d)

df_predictions.to_csv('xgbsubmit.csv', index=False)