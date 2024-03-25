# INTRODUCTION 
About my project ["PREDECTING REAL ESTATE PRICE USING LINEAR REGRESSION AND SUPPORT VECTOR Machine"]
# DOMAIN --->
![image](https://github.com/manikantareddychamala/PREDICTING-REAL-ESTATE-PRICE-A-COMPARATIVE-ANALYSIS-OF-LINEAR-REGRESSION-AND-SUPPORT-VECTOR-MACHINE/assets/162694056/18441df2-27a1-4e02-93a7-0d613fcc299e)!
![image](https://github.com/manikantareddychamala/PREDICTING-REAL-ESTATE-PRICE-A-COMPARATIVE-ANALYSIS-OF-LINEAR-REGRESSION-AND-SUPPORT-VECTOR-MACHINE/assets/162694056/9fbe3996-ab7e-4834-a695-31f7f4ff3a69)! 
![image](https://github.com/manikantareddychamala/PREDICTING-REAL-ESTATE-PRICE-A-COMPARATIVE-ANALYSIS-OF-LINEAR-REGRESSION-AND-SUPPORT-VECTOR-MACHINE/assets/162694056/5db2af12-c4c3-4002-8cd6-3f1de82f0ea0)!  ![image](https://github.com/manikantareddychamala/PREDICTING-REAL-ESTATE-PRICE-A-COMPARATIVE-ANALYSIS-OF-LINEAR-REGRESSION-AND-SUPPORT-VECTOR-MACHINE/assets/162694056/807bec53-6344-4ce3-8697-31fd83968db0)![image](https://github.com/manikantareddychamala/PREDICTING-REAL-ESTATE-PRICE-A-COMPARATIVE-ANALYSIS-OF-LINEAR-REGRESSION-AND-SUPPORT-VECTOR-MACHINE/assets/162694056/673f93ed-9dd8-4efc-9d20-49b146f909e1)![image](https://github.com/manikantareddychamala/PREDICTING-REAL-ESTATE-PRICE-A-COMPARATIVE-ANALYSIS-OF-LINEAR-REGRESSION-AND-SUPPORT-VECTOR-MACHINE/assets/162694056/a48a1cd7-f7b3-4ea4-a31d-b637c96bd74c)![image](https://github.com/manikantareddychamala/PREDICTING-REAL-ESTATE-PRICE-A-COMPARATIVE-ANALYSIS-OF-LINEAR-REGRESSION-AND-SUPPORT-VECTOR-MACHINE/assets/162694056/239ad3c8-11ab-4d67-b46b-f528a5f3b832)







# DATA PREPROCESSING --->

# 1.Need to import libraries and modules in to the program
 
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

# 5. FIT THE LINEAR REGRESSIONFOR TRAING DATA
from sklearn.model import Linearregression;

regressor = LinearRegression;

regressor.fit(x_train,y_train)

# 6. predict the value's that correct output or not
y_predict = regressor.predict(x_test)

x_predict = regressor.predict(y_train)

# 7.  we will go for validation for traing dataset;
mtp.scatter()(x_train,y_train,colour = "Green");

mtp.plot(x_train,y_predict,colour = "Red");

mtp.title("bath, totalsqfoot, bhk vs "prices")

mtp.x_label("Toatlsqfoot, bhk, bath");

mtp.y_label("prices");
# 8. then after we have to do the validation on testing the same instead of x_train,y_train we use x_test, y_test;
conclusion: linear regression is the Best fit model execution


![image](https://github.com/manikantareddychamala/PREDICTING-REAL-ESTATE-PRICE-A-COMPARATIVE-ANALYSIS-OF-LINEAR-REGRESSION-AND-SUPPORT-VECTOR-MACHINE/assets/162694056/c65e6102-d161-4bd6-9433-44212d37b59f) ![image](https://github.com/manikantareddychamala/PREDICTING-REAL-ESTATE-PRICE-A-COMPARATIVE-ANALYSIS-OF-LINEAR-REGRESSION-AND-SUPPORT-VECTOR-MACHINE/assets/162694056/cd6fed58-c523-4bd2-8fe9-4f3e13770392)


# support vector machine process in machine learning
from sklearn.model import svc(classifer)

classifer = support vector machine

classifier.fit(x_train,y_train)

# 2. predict the values
y_train = classier.fit(x_train)

# 3. validation on train set results

# 4. validation on test set results
[
![image](https://github.com/manikantareddychamala/PREDICTING-REAL-ESTATE-PRICE-A-COMPARATIVE-ANALYSIS-OF-LINEAR-REGRESSION-AND-SUPPORT-VECTOR-MACHINE/assets/162694056/7514b35a-d017-4f07-bbc3-67ac81ab6fec)
](url) ![image](https://github.com/manikantareddychamala/PREDICTING-REAL-ESTATE-PRICE-A-COMPARATIVE-ANALYSIS-OF-LINEAR-REGRESSION-AND-SUPPORT-VECTOR-MACHINE/assets/162694056/2bc38d8b-1b12-408a-877f-12d5dbc8cf33) ![image](https://github.com/manikantareddychamala/PREDICTING-REAL-ESTATE-PRICE-A-COMPARATIVE-ANALYSIS-OF-LINEAR-REGRESSION-AND-SUPPORT-VECTOR-MACHINE/assets/162694056/3bfb9ef8-1059-482d-83d3-d96381757fc2)

# 5. CONCLUSION : Finally after doing two Algorithm executions we got the best fit model performance is "SUPPORT VECTOR MACHINE"(SVM) . We got the predected values and actual values are equal in svm model
      
        [    support vector machine accuracy more than the linear regression 
![image](https://github.com/manikantareddychamala/PREDICTING-REAL-ESTATE-PRICE-A-COMPARATIVE-ANALYSIS-OF-LINEAR-REGRESSION-AND-SUPPORT-VECTOR-MACHINE/assets/162694056/bf00cf64-c523-4da3-a50f-875066f295e6)
](url)


