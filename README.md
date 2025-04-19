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

A program to implement the prediction of iris species using an SGD Classifier.
Developed by: B R SWETHA NIVASINI 
RegisterNumber:  212224040345
```

```
 import pandas as pd
 from sklearn.datasets import load_iris
 from sklearn.linear_model import SGDClassifier
 from sklearn.model_selection import train_test_split
 from sklearn.metrics import accuracy_score, confusion_matrix
 import matplotlib.pyplot as plt
 import seaborn as sns
 df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
 df['target'] = iris.target
 print(df.head())
 X = df.drop('target', axis=1)
 y = df['target']
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
 random_state=42)
 sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)
 sgd_clf.fit(X_train, y_train)
 y_pred = sgd_clf.predict(X_test)
 accuracy = accuracy_score(y_test, y_pred)
 print(f"Accuracy: {accuracy:.3f}")
 cm = confusion_matrix(y_test, y_pred)
 print("Confusion Matrix:")
 print(cm)


```

## Output:
![prediction of iris species using SGD Classifier](sam.png)

# HEAD
![Screenshot 2025-04-19 174107](https://github.com/user-attachments/assets/68a07420-6cd2-4f4a-a5f4-84891e257770)

# TARGET
![Screenshot 2025-04-19 174115](https://github.com/user-attachments/assets/c6899cc1-d67c-41b3-a92d-efab7f54094e)

# ACCURACY
![Screenshot 2025-04-19 174124](https://github.com/user-attachments/assets/81fe6a30-6495-4203-a722-a62fb463d444)

# CONFUSION MATRIX
![Screenshot 2025-04-19 174132](https://github.com/user-attachments/assets/ee61a162-3955-4f99-8e05-e7e01d1b0867)







## Result:
Thus, the program to implement the prediction of the Iris species using an SGD Classifier is written and verified using Python programming.
