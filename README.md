# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipment Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.  Import Necessary Libraries and Load Data
2.  Split the Dataset into Training and Testing Sets
3.  Train the Model Using Stochastic Gradient Descent (SGD)
4.  Make Predictions and Evaluate Accuracy
5.   Generate a Confusion Matrix

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: B R SWETHA NIVASINI B R
RegisterNumber:  212224040345
*/
```
```
import pandas as pd 
from sklearn.datasets import load_iris 
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, confusion_matrix 
import matplotlib.pyplot as plt 
import seaborn as sns 
iris=load_iris() 
df=pd.DataFrame(data=iris.data, columns=iris.feature_names) 
df['target']=iris.target 
print(df.head())
```
![image](https://github.com/user-attachments/assets/9de78df2-c732-4f64-8de1-26a5689107e1)

```
X = df.drop('target',axis=1) 
y=df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 )

sgd_clf=SGDClassifier(max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train,y_train)

y_pred=sgd_clf.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.3f}")
```
![image](https://github.com/user-attachments/assets/db71bbe4-93f2-4a96-a3fb-5c6e5c2448c2)

```
cm=confusion_matrix(y_test,y_pred) 
print("Confusion Matrix:") 
print(cm)
```
![image](https://github.com/user-attachments/assets/11b698c0-64d3-415f-b444-aa321bdc3821)

```
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt='d', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
```
![image](https://github.com/user-attachments/assets/e9e02331-0c1c-44d5-8efa-18844c055bcc)







## Result:
Thus, the program to implement the prediction of the Iris species using an SGD Classifier is written and verified using Python programming.
