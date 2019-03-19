import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import Imputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV


df = pd.read_csv('test.csv')

cols_drop = ['surveyid', 'survey_date']

# cols_drop = ['ent_employees','surveyid','survey_date','amount_given_mpesa','amount_received_mpesa',
# 			 'med_vacc_newborns','med_child_check','hh_totalmembers','early_survey','day_of_week','asset_niceroof',
# 			 'wage_expenditures', 'given_mpesa']

df = df.drop(cols_drop, axis=1)

cols_em = df.columns[df.isnull().any()].tolist()

imputer = Imputer(missing_values=np.nan, strategy='mean', axis=1)

df[cols_em] = imputer.fit_transform(df[cols_em])

empty = pd.isnull(df).sum()

df.to_csv('Clean_test.csv', index=False)