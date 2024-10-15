# Machine-Learning
This Repo contains various Algorithms of ML

### Detailed Writeup for the Linear Regression Code with Cross-Validation on the California Housing Dataset

#### **Introduction:**
The aim of this project is to perform a Linear Regression analysis on the California Housing dataset to predict house prices based on various features. We will evaluate the performance of our model using Cross-Validation, focusing specifically on the Mean Squared Error (MSE) metric to assess how well the model performs across different folds of the data.

### **Step-by-Step Explanation:**

#### **1. Importing Libraries:**
We start by importing the necessary libraries:
- `pandas` and `numpy` for data manipulation and calculations.
- `fetch_california_housing` from `sklearn.datasets` to load the dataset.
- `LinearRegression` from `sklearn.linear_model` to apply the linear regression model.
- `cross_val_score` from `sklearn.model_selection` to perform cross-validation.
- `matplotlib.pyplot` for visualizing the results.

```python
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
```

#### **2. Loading the Dataset:**
The California Housing dataset is fetched using the `fetch_california_housing()` function from `sklearn`. The data is loaded into a `pandas` DataFrame for easy handling, and we name the features using the `feature_names` provided in the dataset.

```python
df = fetch_california_housing()
dataset = pd.DataFrame(df.data)
dataset.columns = df.feature_names
dataset['final_val'] = df.target
```
Here, we also add the target variable (house prices) as a new column `final_val` to the DataFrame. 

#### **3. Defining Features (X) and Target (y):**
Next, we separate the independent variables (features) and the dependent variable (target). 
- `X` contains all the feature columns (everything except the last column).
- `y` contains the target column (`final_val`), which we are trying to predict.

```python
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
```

#### **4. Applying Linear Regression:**
Now, we initialize the Linear Regression model and evaluate its performance using Cross-Validation. We use 18-fold cross-validation, which means that the data is split into 18 subsets, and the model is trained on 17 subsets while the remaining one is used for testing. This process is repeated 18 times, and each time, we compute the Mean Squared Error (MSE).

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

lin_reg = LinearRegression()
mse = cross_val_score(lin_reg, X, y, scoring='neg_mean_squared_error', cv=18)
```

**Note**: We use `neg_mean_squared_error` as the scoring parameter because `cross_val_score` expects a maximization score. The negative of the MSE is used so that a higher value indicates better performance. We will convert it back to a positive value later for interpretability.

#### **5. Computing the Mean MSE:**
After performing cross-validation, we take the average of all the MSE scores to get a sense of the overall performance of the model.

```python
mean_mse = np.mean(mse)
```
Since the scores are negative, the mean value will also be negative, but this simply indicates that we are working with an error metric (lower values are better).

#### **6. Visualizing the Results:**
Finally, we plot the MSE values for each of the 18 cross-validation folds. We convert the negative MSE values to positive for easier understanding.

```python
import matplotlib.pyplot as plt

mse_positive = -mse

plt.figure(figsize=(10,6))
plt.bar(range(1, len(mse_positive) + 1), mse_positive)
plt.xlabel('Cross-Validation Fold')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('MSE for Each Fold in Cross-Validation')
plt.show()
```
This bar chart provides a clear visualization of the MSE values for each fold, showing how the model performed across different subsets of the data.

![image](https://github.com/user-attachments/assets/3b6bf9ed-7a50-4499-8e2f-a951ae85afa8)


### **Conclusion:**
The objective of this exercise was to evaluate the performance of a Linear Regression model on the California Housing dataset using cross-validation. By visualizing the MSE for each fold, we can observe the consistency of the model’s performance across different parts of the data.

- **The key takeaway**: The model's performance, as indicated by the MSE values, varies slightly across different folds, but overall, the results help us gauge the reliability of the model. Lower MSE values indicate better predictive performance, meaning that the model is making smaller errors in predicting house prices.
  
This method helps prevent overfitting and gives us a better understanding of how the model would perform on unseen data, making it a valuable tool in machine learning.
