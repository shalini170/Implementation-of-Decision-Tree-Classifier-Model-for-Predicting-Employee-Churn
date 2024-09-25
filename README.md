# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.
2.Upload the dataset and check for any null values using .isnull() function.
3.Import LabelEncoder and encode the dataset.
4.Import DecisionTreeClassifier from sklearn and apply the model on the dataset.
5.Predict the values of array.
6.Import metrics from sklearn and calculate the accuracy of the model on the dataset. 
7.Predict the values of array. 8.Apply to new unknown values.
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Shruthi.S
RegisterNumber:  212222220044
*/
```
```
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics   
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
### DATA HEAD
![x1](https://user-images.githubusercontent.com/94154683/169513081-6f903385-3da1-4ba6-ae1e-27cb76d591a6.png)
### DATA INFO
![x3](https://user-images.githubusercontent.com/94154683/169513082-47b67344-6c44-4ff8-a520-8c341f312eab.png)
### DATA ISNULL
![x3as3](https://user-images.githubusercontent.com/94154683/169513065-14c88236-9210-487c-a477-78c487c4489a.png)
### DATA LEFT
![x5](https://user-images.githubusercontent.com/94154683/169513070-33d9ed74-b4f7-4e46-9966-8c60efbec5e5.png)
### X HEAD
![x6](https://user-images.githubusercontent.com/94154683/169513073-402a0256-0d5a-46d6-a34a-271cd223be71.png)
### DATA FIT
![x7](https://user-images.githubusercontent.com/94154683/169513077-04505194-c763-4523-9d17-fbff648e94dd.png)
### ACCURACY
![x8](https://user-images.githubusercontent.com/94154683/169513079-42266498-9323-4a5a-87d6-94ba3815d40b.png)
### PREDICTED VALUES
![XX](https://user-images.githubusercontent.com/94154683/169514059-120bf7f7-e57d-4726-b459-4d9a39cdfd2d.png)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
