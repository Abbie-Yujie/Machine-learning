# Machine-learning
# Predict movie box office
####################################
######## Data clean ##################
####################################

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import math
# 1.1 read data
df=pd.read_csv(r"PMA_blockbuster_movies.csv")
# 1.2 delete all empty rows 
df.dropna(axis=0,how='all',inplace=True)
# 1.3 delete useless columns
df.drop(columns=['poster_url','genres','title','release_date'],inplace=True)
# 1.4 delete duplicate rows
df=df.drop_duplicates()
df.info()
# 1.5 pandas.get_dummies
df = pd.get_dummies(df, columns=["Genre_1","Genre_2","Genre_3","rating",'studio'], prefix=["Genre_1","Genre_2","Genre_3","rating",'studio'])
df.shape
### (397,171)
# 1.6 move the "adjusted" column to the first column
data=df['adjusted']
df.drop(labels=['adjusted'], axis=1,inplace = True)
df.insert(0, 'adjusted', data)
# 1.7 check missing value
df.isna().sum().head()
# because there are not missing value, there is no need to fill the missing values
'''
adjusted             0
rt_audience_score    0
rt_freshness         0
2015_inflation       0
imdb_rating          0
dtype: int64
'''
# 1.8 converting currency with $ to float 
# source: https://stackoverflow.com/questions/32464280/converting-currency-with-to-numbers-in-python-pandas
df[df.columns[0]] = df[df.columns[0]].replace('[\$,]', '', regex=True).astype(float)
df[df.columns[6]] = df[df.columns[6]].replace('[\$,]', '', regex=True).astype(float)

# 1.9 covert "2015_inflation" into float
# https://www.geeksforgeeks.org/python-pandas-series-to_frame/
# https://blog.csdn.net/liangyingyi1006/article/details/77644432
from pandas import DataFrame
df['2015_inflation'] = df['2015_inflation'].str.strip("%").astype(float)/100;

# 1.10 Normalization
from sklearn.preprocessing import StandardScaler
x_list = df.columns.tolist()
clf_std=StandardScaler()
df[x_list]=clf_std.fit_transform(df[x_list])
df.head()
#1.11 read out new dataset
df.to_csv("output.csv")

#1.12 PCA
df1=df.drop(['adjusted'],axis=1)
from sklearn.decomposition import PCA
model_pca = PCA(n_components=30)
model_pca.fit(df1)
df1 = pd.DataFrame(data=df1,index=df1.index)
df=df['adjusted'].to_frame().join(df1,how='right')

# 1.13 split data into training and test 
x= df.drop(['adjusted'],axis=1)
y= df['adjusted']
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size = 0.2, random_state=5) 
# print the shapes to check everything is OK
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
'''
(317, 170)
(80, 170)
(317,)
(80,)
'''

####################################
######## Decision Tree #############
####################################
## 2.1 for discrete numbers: DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor as DTR
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
dtr = DTR()
dtr_fit = dtr.fit(X_train, Y_train)
y_predicted = dtr.predict(X_test)
#rms = math.sqrt(mean_squared_error(Y_test, y_predicted))
#print("rms error is: " + str(rms))
print("R2_score: " + str(r2_score(Y_test, y_predicted)))
'''R2_score: 0.7206863614826868 '''

# 2.2 Optimise hyperparameters
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.metrics import r2_score

tuned_parameters = [{'criterion': ['mse', 'friedman_mse','mae'],
                     'max_depth': [3, 5, 7],
                     'min_samples_split': [3, 5, 7],
                     'min_samples_leaf':[3, 5, 7],
                     'max_features': ["sqrt", "log2", None]}]

scores = ['r2']

for score in scores:
    print("# Tuning hyperparameters for %s" % score)
    print("\n")
    clf = GridSearchCV(DTR(), tuned_parameters, cv=5,
                       scoring= score)
    clf.fit(X_train, Y_train)
    print("Best parameters set found on the training set:")
    print(clf.best_params_)
    print("\n")

    '''
Best parameters set found on the training set:
{'criterion': 'mae', 'max_depth': 7, 'max_features': None, 'min_samples_leaf': 3, 'min_samples_split': 7}
 '''
# 2.3 Decision Tree Regressor after optimising hyperparameters
dtr = DTR(criterion='mae', max_depth=7, max_features=None, min_samples_leaf=3, min_samples_split=7)
dtr_fit = dtr.fit(X_train, Y_train)
y_predicted = dtr.predict(X_test)
#rms = math.sqrt(mean_squared_error(Y_test, y_predicted))
#print("rms error is: " + str(rms))
print("DTR_R2_score: " + str(r2_score(Y_test, y_predicted)))

'''DTR_R2_score: 0.807290082916448 '''

####################################
##Support Vector Regressor (SVR) ###
####################################
# 3.1 SVR
# source: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from math import sqrt

svr = SVR()
svr_fit=svr.fit(X_train,Y_train)
# y_predicted = svr.predict(X_test)
# print("R2_sore: " + str(r2_score(Y_test, y_predicted)))
svr_fit.score(X_train,Y_train)
'''R2_score: 0.79426938162291'''

# 3.2 Optimise hyperparameters
tuned_parameters = [{ 'C': [1, 10, 100, 100],
                     'kernel': ['rbf'],
                     'degree': [3, 5, 7],
                     'gamma': ['scale', 'auto',3, 5, 7]}]

scores = ['r2']

for score in scores:
    print("# Tuning hyperparameters for %s" % score)
    print("\n")
    clf = GridSearchCV(SVR(), tuned_parameters, cv=5,
                       scoring= score)
    clf.fit(X_train, Y_train)
    print("Best parameters set found on the training set:")
    print(clf.best_params_)
    print("\n")
'''
Best parameters set found on the training set:
{'C': 100, 'degree': 3, 'gamma': 'auto', 'kernel': 'rbf'}

'''

# 3.3 Support Vector Regressor after optimising hyperparameters
svr = SVR(C=100, degree=3, gamma='auto', kernel='rbf')
svr_fit=svr.fit(X_train,Y_train)
y_predicted = svr.predict(X_test)
print("SVR_R2_score: " + str(r2_score(Y_test, y_predicted)))
'''SVR_R2_score: 0.8629199155713818 '''

####################################
######RandomForestRegressor#########
####################################
# 4.1 source:https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html?highlight=randomforest#sklearn.ensemble.RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.datasets import make_regression
regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(X_train,Y_train)
y_predicted = regr.predict(X_test)
print("R2_score: " + str(r2_score(Y_test, y_predicted)))
'''R2_score: 0.7650976677495838'''

# 4.2 Optimise hyperparameters

tuned_parameters = [{ 'n_estimators': [10, 20, 30],
                     'max_depth': [2,3, 5, 7,11,None],
                     'max_features': ['auto', 'sqrt','log2', 3, 5, 7]}]

scores = ['r2']
for score in scores:
    print("# Tuning hyperparameters for %s" % score)
    print("\n")
    clf = GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=5,
                       scoring= score)
    clf.fit(X_train, Y_train)
    print("Best parameters set found on the training set:")
    print(clf.best_params_)
    print("\n")
'''
Best parameters set found on the training set:
{'max_depth': 11, 'max_features': 'auto', 'n_estimators': 20}
'''
#4.3 RandomForestRegressor after optimising hyperparameters
from sklearn.ensemble import RandomForestRegressor 
from sklearn.datasets import make_regression
regr = RandomForestRegressor(max_depth=11, random_state=0, max_features="auto", n_estimators =20)
regr_fit=regr.fit(X_train,Y_train)
y_predicted = regr.predict(X_test)
print("RFR_R2_score: " + str(r2_score(Y_test, y_predicted)))

'''RFR_R2_score: 0.8297926299766221 '''

####################################
###########Conclusion###############
####################################
## Because the R2_score of SVR is the most higher one, which present higher accuracy, so we recommend model 2: SVR to identify the success of a movie.
## we also recommend use Voting Regressor from https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html?highlight=voting#sklearn.ensemble.VotingRegressor to optimize the result of prediction among these three models.
