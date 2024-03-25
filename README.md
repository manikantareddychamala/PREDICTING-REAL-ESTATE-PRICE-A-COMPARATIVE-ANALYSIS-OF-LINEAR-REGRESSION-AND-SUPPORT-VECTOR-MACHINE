#DATA PREPROCESSING
#1.Need to import libraries and modules in to the program
 
  import numpy as nm
  import matplotlib.pyplot as mtp
  import pandas as pd
  
# 2.we have to load the dataset
data_set = pd.read_cv("BENGULAR_DATASET")

# 3. Extract the dependent and independent variables into data set for location(iloc: stands for index location)
x = data_set.iloc[::-1]

y = data_set.iloc[::1]
# 4. split the dataset into two parts for testing and training
#sklearn.library

from sklearn.model_selection import train_test_spilt:
x_train,x_test,y_test,y_train = train_test_split(x,y,test_size=0.3, random_state=5)

FIT THE LINEAR REGRESSIONFOR TRAING DATA
from sklearn.model import Linearregression;
regressor = LinearRegression;
regressor.fit(x_train,y_train)
#predict the value's that correct output or not
y_predict = regressor.predict(x_test)
x_predict = regressor.predict(y_train)
# we will go for validation for traing dataset;
mtp.scatter()(x_train,y_train,colour = "Green");
mtp.plot(x_train,y_predict,colour = "Red");
mtp.title("bath, totalsqfoot, bhk vs "prices")
mtp.x_label("Toatlsqfoot, bhk, bath");
mtp.y_label("prices");
#tehn after we have to do the validation on testing the same instead of x_train,y_train we use x_test, y_test;
conclusion: linear regression is the Best fit model execution
