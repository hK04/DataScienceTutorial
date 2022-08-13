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
Not the best, but at least existing solution to Titanic Kaggle problem. Solution based on Logistic Regression with `ElasticNet`. Coefficients were find through self-implemented values_grid, so the result may be improved with change of model, on that ML based, or with more detail work with data. Maybe 'Rich' feature isn't that good, as it was predicted. Or maybe, i shouldn't thrown away all NaN data ;)))

Implementation in `kaggle\titanic\main.ipynb`, solution in `submit.csv`

### 2. Prediction of the Churn of clients
![image](https://user-images.githubusercontent.com/65892626/184506301-ade5fa73-99f0-4038-bba1-3b433d6dfca5.png)

Algorithm based on comparison of **Random Forest**, **CatBoost** and **LogisticRegression**
<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/65892626/184506171-1fc1c2c2-8bc9-4fb6-9f68-e3e9c19df4ec.png">
</p>

Performance  may be increased with rebalancing the `train.csv`

Implementation in `kaggle\PredictionOfTheChurnOfClients\main.ipynb`, solution in `submit.csv`
