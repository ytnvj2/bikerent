import os
import pandas as pd
import numpy as np
import fancyimpute
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (explained_variance_score, mean_absolute_error, mean_squared_error)
from sklearn.metrics import r2_score
def load_data():
    # define path of csv
    raw_data_path=os.path.join('day.csv')
    # import the dataset
    df=pd.read_csv(raw_data_path)
    df.columns=['Instant', 'Date', 'Season', 'Year', 'Month', 'Holiday', 'Weekday',
           'WorkingDay', 'WeatherSituation', 'Temp', 'ActualTemp', 'Humidity', 'WindSpeed',
           'Casual', 'Registered', 'Count']
    # Convert the categorical columns to object 
    cat_cols=['Instant','Season', 'Year', 'Month', 'Holiday', 'Weekday',
           'WorkingDay', 'WeatherSituation']
    for i in cat_cols:
        df[i] = df[i].astype(object)
    num_cols=[]
    for i in df.columns:
        if(df[i].dtype==np.dtype('int64') or df[i].dtype==np.dtype('float64')):
            num_cols.append(i)
            df[i] = df[i].astype(np.float64)
    # viewing the dataframe's info
    df.info()
    return df,num_cols,cat_cols
def outlier_imputer(df_o,num_cols):
    # Outlier Analysis
    while True:
        for i in num_cols:
            min=(df_o[i].quantile(0.25)-1.5*(df_o[i].quantile(0.75)-df_o[i].quantile(0.25)))    
            max=(df_o[i].quantile(0.75)+1.5*(df_o[i].quantile(0.75)-df_o[i].quantile(0.25)))
            df_o.loc[df_o[i]<min,i] = np.nan
            df_o.loc[df_o[i]>max,i] = np.nan
        missing_val = df_o.isnull().sum()
        print(missing_val)
        if(missing_val.sum()>0):
            df_o_knn=pd.DataFrame(fancyimpute.KNN(k = 3).complete(df_o[num_cols]), columns = num_cols)
            df_o_knn.head()
            df_o.iloc[:,9:15]=df_o_knn.iloc[:,:]
        else:
            break
    return df_o
def feature_selection(df):
    #Set the width and hieght of the plot
    f, ax = plt.subplots(figsize=(7, 5))
    #Generate correlation matrix
    corr = df.iloc[:,9:].corr()
    #Plot using seaborn library
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True, ax=ax)
    return df.drop(['Instant','Date','Holiday','Temp','Registered'],axis=1,inplace=False)
def split_dataset(df):
    X=df.iloc[:,:-1].values
    y=df.iloc[:,-1].values
    return train_test_split(X,y,test_size=0.2,random_state=123)
def feature_scaling(X_train,X_test):
    standardScaler=StandardScaler()
    X_train[:,6:]=standardScaler.fit_transform(X_train[:,6:])
    X_test[:,6:]=standardScaler.transform(X_test[:,6:])
    return X_train,X_test,standardScaler
def train_lm(X_train,y_train):
    lr_model=LinearRegression()
    lr_model.fit(X_train,y_train)
    return lr_model
def train_dt(X_train,y_train):
    dtr_model=DecisionTreeRegressor(random_state=123)
    dtr_model.fit(X_train,y_train)
    return dtr_model
def train_rf(X_train,y_train):
    rf_model=RandomForestRegressor(n_estimators=50,random_state=123)
    rf_model.fit(X_train,y_train)
    return rf_model
def train_knn(X_train,y_train):
    rf_model=KNeighborsRegressor(n_neighbors=5)
    rf_model.fit(X_train,y_train)
    return rf_model
def predict_vals(model,X_test,y_test):
    print(model.score(X_test,y_test))
    preds=model.predict(X_test)
    return preds
def evaluate_model(y_test,y_pred):
    print('R-Square',r2_score(y_test,y_pred))
    print('MSE',mean_squared_error(y_test,y_pred))
    print('MAE',mean_absolute_error(y_test,y_pred))
    print('Explained Variance',explained_variance_score(y_test,y_pred))
def predict(model,X):
    df=X
    df.columns=['Instant', 'Date', 'Season', 'Year', 'Month', 'Holiday', 'Weekday',
           'WorkingDay', 'WeatherSituation', 'Temp', 'ActualTemp', 'Humidity', 'WindSpeed',
           'Casual', 'Registered']
    # Convert the categorical columns to object 
    cat_cols=['Season', 'Year', 'Month', 'Holiday', 'Weekday',
           'WorkingDay', 'WeatherSituation']
    for i in cat_cols:
        df[i] = df[i].astype(object)
    num_cols=[]
    for i in df.columns:
        if(df[i].dtype==np.dtype('int64') or df[i].dtype==np.dtype('float64')):
            num_cols.append(i)
            df[i] = df[i].astype(np.float64)
    X.drop(['Instant','Date','Holiday','Temp','Registered'],axis=1,inplace=True)
    X=X.values
    X[:,6:]=standardScaler.transform(X[:,6:])
    return model.predict(X)
df,num_cols,cat_cols=load_data()
df=outlier_imputer(df,num_cols)
df=feature_selection(df)
X_train,X_test,y_train,y_test=split_dataset(df)
X_train_scaled,X_test_scaled,standardScaler=feature_scaling(X_train,X_test)
lr_model=train_lm(X_train,y_train)
dt_model=train_dt(X_train,y_train)
rf_model=train_rf(X_train,y_train)
knn_model=train_knn(X_train,y_train)
print('Linear Model')
y_pred=predict_vals(lr_model,X_test,y_test)
evaluate_model(y_test,y_pred)
print('Decision Tree Model')
y_pred=predict_vals(dt_model,X_test,y_test)
evaluate_model(y_test,y_pred)
print('Random Forest Model')
y_pred=predict_vals(rf_model,X_test,y_test)
evaluate_model(y_test,y_pred)
print('K Nearest Neighbors(k=5)')
y_pred=predict_vals(knn_model,X_test,y_test)
evaluate_model(y_test,y_pred)
# Sample Input Creation
a=np.array([[399, '2012-02-03', 1, 1, 2, 0, 5, 1, 1, 0.313333, 0.309346,
        0.526667, 0.17849600000000002, 310, 3841]], dtype=object)
s=pd.DataFrame(a)
print(s)
# Output for sample input
print("The Model's Prediction for the input is ",predict(rf_model,s))