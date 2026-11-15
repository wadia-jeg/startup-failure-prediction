import re
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, roc_auc_score, mutual_info_score, root_mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

data = 'data/Startup Failures.csv'

df = pd.read_csv(data)


df.columns =  df.columns.str.lower().str.replace(' ','_')
df.head()


def clean_how_much_they_raised(text):
    number = pd.to_numeric(re.sub(r'[^\d.]','', str(text)), errors='coerce')
    
    if 'B' in str(text):
        return number * 1_000_000_000
    
    if 'M' in str(text):
        return number * 1_000_000

    if 'K' in str(text):
        return number * 1_000

    return 0

df['total_funding'] = df.how_much_they_raised.apply(clean_how_much_they_raised)

df.total_funding = np.log1p(df.total_funding )


def clean_years_of_operation(text):
    years = re.findall(r'\d{4}', str(text))

    return int(years[1]) - int(years[0])

df.years_of_operation = df.years_of_operation.apply(clean_years_of_operation)

df = df.fillna(0)



categorical = ['sector','giants',
       'no_budget', 'competition', 'acquisition_stagnation',
       'platform_dependency', 'monetization_failure', 'niche_limits',
       'execution_flaws', 'trend_shifts', 'toxicity/trust_issues',
       'regulatory_pressure', 'overhype']

numerical = ['years_of_operation', 'total_funding']

## Classification

class_features = ['sector','competition', 'giants' ,'niche_limits' ,'monetization_failure', 'execution_flaws', 'acquisition_stagnation', 'years_of_operation']
classj_y = df.poor_market_fit

df_train_full, df_test, y_train_full, y_test = train_test_split(df[class_features], classj_y, test_size=0.2, random_state=1)
df_train, df_val, y_train, y_val = train_test_split(df_train_full, y_train_full, test_size=0.25 ,random_state=1)

class_vectorizer = DictVectorizer(sparse=False)

train_dicts = df_train.to_dict(orient='records')
val_dicts = df_val.to_dict(orient='records')


X_train = class_vectorizer.fit_transform(train_dicts)
X_val = class_vectorizer.transform(val_dicts)


class_model = LogisticRegression(C=10, max_iter=200)
class_model.fit(X_train, y_train)

y_pred = class_model.predict_proba(X_val)[:,1]

print(roc_auc_score(y_val, y_pred))
## Regression 

reg_features = ['sector','competition', 'giants' ,'niche_limits' ,'monetization_failure', 'execution_flaws', 'acquisition_stagnation', 'years_of_operation']
reg_y = df.total_funding

df_reg_train_full, df_test, y_train_full, y_test = train_test_split(df[reg_features], reg_y, test_size=0.2, random_state=1)
df_train, df_val, y_train, y_val = train_test_split(df_reg_train_full, y_train_full, test_size=0.25 ,random_state=1)

reg_vectorizer = DictVectorizer(sparse=False)

train_dicts = df_train.to_dict(orient='records')
val_dicts = df_val.to_dict(orient='records')


X_train = reg_vectorizer.fit_transform(train_dicts)
X_val = reg_vectorizer.transform(val_dicts)

reg_model = RandomForestRegressor()
reg_model.fit(X_train, y_train)

y_pred = reg_model.predict(X_val)

print(root_mean_squared_error(y_val, y_pred))

with open('models.bin', "wb") as file:
    pickle.dump((class_vectorizer, class_model, reg_vectorizer, reg_model), file)