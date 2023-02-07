FINAL PROJECT
SALES ANALYSIS
Author: JESSICA URIBE

The goal of this is to help the retailer understand the properties of products and outlets that play crucial roles in predicting sales.

Your first task is to build a linear regression model to predict sales.
Build a linear regression model. Evaluate the performance of your model based on r^2. Evaluate the performance of your model based on rmse.

Your second task is to build a regression tree model to predict sales.
Build a simple regression tree model. Compare the performance of your model based on r^2. Compare the performance of your model based on rmse.

You now have tried 2 different models on your data set. You need to determine which model to implement.
Overall, which model do you recommend? Justify your recommendation.

To finalize this project, complete a README in your GitHub repository including:
An overview of the project 2 relevant insights from the data (supported with reporting quality visualizations) Summary of the model and its evaluation metrics Final recommendations
Data:
df = pd.read_csv('/content/drive/MyDrive/Coding Dojo/sales_predictions.csv')
df.head()

Methods
# Imports
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, \
OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn import set_config
set_config(display='diagram')



Visual 2 Title
Model
Describe your final model

Report the most important metrics

Refer to the metrics to describe how well the model would solve the business problem

Recommendations:
More of your own text here

Limitations & Next Steps
More of your own text here

For further information
For any additional questions, please contact email
