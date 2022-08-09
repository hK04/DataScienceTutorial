# DataScienceTutorial
Bunch of Notebooks filled with information about DS libraries, methods and implementations studied from [DeepLearningSchool](https://www.dlschool.org/) and other sources 
## 1. Algoritms: 
### 1.1 Linear Regression:
![LinearRegressionPicture](https://user-images.githubusercontent.com/65892626/183291212-7bafd970-b53f-4cf9-aea5-e58334139e03.png)
Basic implementation in `LinearRegression.ipynb`.
Also i implemented **Ridge** and **Lasso** regularization, random Batch without division on epochs
### 1.2. Logistic Regression: 
![LogisticRegression](https://user-images.githubusercontent.com/65892626/183291276-842ba2c9-0f28-4d1f-b018-93d3dd850074.png)

Basic implementation in `LogisticRegression.ipynb`
## 2. Real datasets approach:
### 1. Titanic
![image](https://user-images.githubusercontent.com/65892626/183523734-30ccf7c9-6ee6-4e57-b0aa-cc9087488bf6.png)
Not the best, but at least existing solution to Titanic Kaggle problem. Solution based on Logistic Regression with Ridge and Lasso. Coefficients were set to 1, so the result may be improved with changing the coeffient to more proper values. Also it's possible, that change of `threshold` may affect perfomance of algorithm

Implementation in `kaggle\titanic\main.ipynb`, solution in `result.csv`

