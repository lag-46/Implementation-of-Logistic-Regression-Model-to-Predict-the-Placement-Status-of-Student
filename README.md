# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import necessary libraries (pandas, LabelEncoder, train_test_split, etc.).

2.Load the dataset using pd.read_csv().

3.Create a copy of the dataset and drop unnecessary columns (sl_no, salary).

4.Check for missing and duplicate values using isnull().sum() and duplicated().sum().

5.Encode categorical variables using LabelEncoder() to convert them into numerical values.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: PANDEESWARAN N
RegisterNumber:  212224230191
*/
```
```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


data = pd.read_csv("C:\\Users\\admin\\Downloads\\Placement_Data.csv")
data.head()

datal = data.copy()
datal = datal.drop(["sl_no", "salary"], axis=1)
datal.head()

print("Missing values:\n", datal.isnull().sum())
print("Duplicate rows:", datal.duplicated().sum())

print("Print Data:")
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
datal["gender"] = le.fit_transform(datal["gender"])
datal["ssc_b"] = le.fit_transform(datal["ssc_b"])
datal["hsc_b"] = le.fit_transform(datal["hsc_b"])
datal["hsc_s"] = le.fit_transform(datal["hsc_s"])
datal["degree_t"] = le.fit_transform(datal["degree_t"])
datal["workex"] = le.fit_transform(datal["workex"])
datal["specialisation"] = le.fit_transform(datal["specialisation"])
datal["status"] = le.fit_transform(datal["status"])
datal

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

print("Y_Prediction array:")
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy value:")
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test, y_pred)
print("Confusion array:")
confusion

from sklearn.metrics import classification_report
classification_reportl = classification_report(y_test, y_pred)
print("Classification Report:\n")
print(classification_reportl)

print("Prediction of LR:")
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])



```

## Output:

## HEAD

<img width="1216" height="263" alt="image" src="https://github.com/user-attachments/assets/ed077820-713b-4901-9abc-179a25518e0a" />

## COPY

<img width="1102" height="277" alt="image" src="https://github.com/user-attachments/assets/42982eff-9a7c-47ea-9d41-c02d0848e335" />

## Missing values,Duplicate rows

<img width="800" height="379" alt="image" src="https://github.com/user-attachments/assets/41345f96-98c0-41a3-9972-65a8a5a5dc62" />

## FIT TRANSFORM

<img width="1074" height="692" alt="image" src="https://github.com/user-attachments/assets/ca6b071e-d1a8-46bc-b3aa-a619e2c9745a" />

## LOGISTIC REGRESSION

<img width="1159" height="349" alt="image" src="https://github.com/user-attachments/assets/abbfa0b4-e87d-4e69-b943-1dde6eb6e312" />

## ACCURACY SCORE

<img width="592" height="170" alt="image" src="https://github.com/user-attachments/assets/9b957f73-f91d-4c5a-8868-75c438211e29" />

## CONFUSION MATRIX

<img width="570" height="203" alt="image" src="https://github.com/user-attachments/assets/0b1e4051-d9a3-446d-b502-e4f10cc6ccaf" />

## CLASSIFICATION REPORT & PREDICTION

<img width="1235" height="562" alt="image" src="https://github.com/user-attachments/assets/70c24331-c60b-44ea-baee-554171b419ca" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
