
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import Imputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV


df = pd.read_csv('train.csv')

cols_drop = ['ent_employees','surveyid','survey_date','amount_given_mpesa','amount_received_mpesa',
			 'med_vacc_newborns','med_child_check','hh_totalmembers','early_survey','day_of_week']

df = df.drop(cols_drop, axis=1)

# df = df.dropna(axis=1, thresh=len(df)*0.6)

df = df.loc[:, df.isnull().mean()<0.6]
cols_em = df.columns[df.isnull().any()].tolist()


imputer = Imputer(missing_values=np.nan, strategy='mean', axis=1)

df[cols_em] = imputer.fit_transform(df[cols_em])

empty = pd.isnull(df).sum()

df.to_csv('Clean_train.csv', index=False)

Y = df['depressed']
X=df.drop(['depressed'], axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.33, random_state=42)

parameters = {'kernel': ('linear', 'rbf'), 'C':[1,10]}

svc = SVC(gamma='auto')
model = GridSearchCV(svc,parameters, cv=5)
model.fit(X_train,Y_train)
joblib.dump(model, 'gridcv.joblib')

predictions = model.predict(X_test)

print(accuracy_score(Y_test,predictions))









