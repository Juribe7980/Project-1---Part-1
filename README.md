Sales Predictions- Final

Jessica Uribe


This week, you will finalize your sales prediction project. The goal of this is to help the retailer understand the properties of products and outlets that play crucial roles in predicting sales.

1. Your first task is to build a linear regression model to predict sales.

Build a linear regression model.
Evaluate the performance of your model based on r^2.
Evaluate the performance of your model based on rmse.
2. Your second task is to build a regression tree model to predict sales.

Build a simple regression tree model.
Compare the performance of your model based on r^2.
Compare the performance of your model based on rmse.
3. You now have tried 2 different models on your data set. You need to determine which model to implement.

Overall, which model do you recommend?
Justify your recommendation.
4. To finalize this project, complete a README in your GitHub repository including:

An overview of the project
2 relevant insights from the data (supported with reporting quality visualizations)
Summary of the model and its evaluation metrics
Final recommendations 
Here is a template you can use for your readme if you would like. You can look at the raw readme file to copy it if you want.

Please note:

Do not include detailed technical processes or code snippets in your README. If readers want to know more technical details they should be able to easily find your notebook to learn more.
Make sure your GitHub repository is organized and professional. Remember, this should be used to showcase your data science skills and abilities.
train test split Data

Load Library and data

## Pandas
import pandas as pd
## Numpy
import numpy as np
## MatPlotLib
import matplotlib.pyplot as plt

## Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer

## Models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor

## Regression Metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

## Tree Model Visualization
from sklearn.tree import plot_tree

## Set global scikit-learn configuration 
from sklearn import set_config
## Display estimators as a diagram
set_config(display='diagram') # 'text' or 'diagram'}

Load the Data

df = pd.read_csv('/content/drive/MyDrive/Coding Dojo/sales_predictions.csv')
df.head()

df.copy()

## Display the number of rows and columns for the dataframe
df.shape
print(f'There are {df.shape[0]} rows, and {df.shape[1]} columns.')
print(f'The rows represent {df.shape[0]} observations, and the columns represent {df.shape[1]-1} features and 1 target variable.')

## Display the column names and datatypes for each column
## Columns with mixed datatypes are identified as an object datatype
df.dtypes

## Display the column names, count of non-null values, and their datatypes
df.info()

Missing Values

## Display the descriptive statistics for the numeric columns
df.describe(include="number") # or 'object'

## Display the descriptive statistics for the non-numeric columns
df.describe(exclude="number") # or 'object'

Dropping a Single Column
The Item Identifier column has high cardinality.
Drop the Item IDentifier column

df.drop(columns = 'Item_Identifier', inplace = True)

df.head()


## Display the descriptive statistics for the non-numeric columns
df.describe(include="number")

No unsual valies noted

print(f'There are {df.duplicated().sum()} duplicate rows.')

#Identify rows, columns and missing values
df.info()

#Display descriptive s tatitistics for all colums
df.describe(include='all')

Split Data (validation Split)

# split X and y, you are predicting price
target = 'Item_Outlet_Sales'

X = df.drop(columns=target).copy()
y = df[target].copy()
X.head()

# Perfoming a train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)
X_train.shape

display(X_train.info())
X_train.head()

num_cols = ['Item_Weight','Item_Visibility','Item_MRP', 'Outlet_establisment_Year']
cat_cols = ['Item_Identifier','Item_Fat_Content','Item_type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']

# Instantiate the transformers
scaler = StandardScaler()
mean_imputer = SimpleImputer(strategy='mean')
freq_imputer = SimpleImputer(strategy='most_frequent')
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
# Prepare separate processing pipelines for numeric and categorical data
num_pipe = make_pipeline(mean_imputer, scaler)
cat_pipe = make_pipeline(freq_imputer, ohe)
# Create ColumnSelectors for the the numeric and categorical data
cat_selector = make_column_selector(dtype_include='object')
num_selector = make_column_selector(dtype_include='number')
# Combine the Pipelines and ColumnSelectors into tuples for the ColumnTransformer
cat_tuple = (cat_pipe, cat_selector)
num_tuple = (num_pipe, num_selector)
# Create the preprocessing ColumnTransformer
preprocessor = make_column_transformer(cat_tuple, num_tuple, remainder='drop')
preprocessor

Linear Regression


Create to predict sales


## Create a function to take the true and predicted values
## and print MAE, MSE, RMSE, and R2 metrics
def model_metrics(pipe, x_train, y_train, x_test, y_test, 
                       model_name='Regression Model', ):
  ## Train
  mae = round(mean_absolute_error(y_train, pipe.predict(x_train)),4)
  mse = round(mean_squared_error(y_train, pipe.predict(x_train)),4)
  rmse = round(np.sqrt(mean_squared_error(y_train, pipe.predict(x_train))),4)
  r2 = round(r2_score(y_train, pipe.predict(x_train)),7)
  print(f'{model_name} Train Scores')
  print(f'MAE: {mae} \nMSE: {mse} \nRMSE: {rmse} \nR2: {r2}\n')

  ## Test
  mae = round(mean_absolute_error(y_test, pipe.predict(x_test)),4)
  mse = round(mean_squared_error(y_test, pipe.predict(x_test)),4)
  rmse = round(np.sqrt(mean_squared_error(y_test, pipe.predict(x_test))),4)
  r2 = round(r2_score(y_test, pipe.predict(x_test)),7)

  ## Display the metrics for the model
  print(f'{model_name} Test Scores')
  print(f'MAE: {mae} \nMSE: {mse} \nRMSE: {rmse} \nR2: {r2}\n')

## Make and fit a linear regression model
reg = LinearRegression()
reg_pipe = make_pipeline(preprocessor, reg)



Make predictions using the testing data

# Fit the model pipeline on the training data
reg_pipe.fit(X_train, y_train)
# Make predictions using the training and testing data
training_predictions = reg_pipe.predict(X_train)
test_predictions = reg_pipe.predict(X_test)
training_predictions[:10]



Make predictions using the testing data.

predictions = reg_pipe.predict(X_test)

prediction_df = X_test.copy()
prediction_df['True Median Price'] = y_test
prediction_df['Predicted Median Price'] = predictions
prediction_df['Error'] = predictions - y_test
prediction_df.head()

Evaluate the performance of your model based on RMSE and r^2.

#Calculating RMSE
train_RMSE = np.sqrt(np.mean(np.abs(training_predictions - y_train)**2))
test_RMSE= np.sqrt(np.mean(np.abs(test_predictions- y_test)**2))
print(f'Model Training RMSE: {train_RMSE}')
print(f'Model Testing RMSE: {test_RMSE}')

#Calculating r2
train_r2= r2_score(y_train, training_predictions)
test_r2 = r2_score(y_test, test_predictions)

print(f'Model Training r2: {train_r2}')
print(f'Model Testing r2: {test_r2}')

DECISION TREE

#Create an instance of the model
dec_tree = DecisionTreeRegressor(random_state = 42)

dec_tree = DecisionTreeRegressor(random_state=42)
dec_tree_pipe = make_pipeline(preprocessor, dec_tree)

#Fit using training data
dec_tree_pipe.fit(X_train, y_train)

model_metrics(dec_tree_pipe, x_train=X_train, y_train=y_train, 
                          x_test=X_test, y_test=y_test, 
                           model_name='Decision Tree Model')

## Display the list of available hyperparameters for tuning
dec_tree.get_params()

## Obtain the max_depth from the pipeline and assign it to the variable max_depth
max_depth = dec_tree_pipe['decisiontreeregressor'].get_depth()
## Display max_depth
max_depth

## Create a range of values from 1 to max_depth to evaluate
depths = range(1, max_depth+1)

## Create a dataframe to store Train and Test  R2 scores
scores = pd.DataFrame(columns=['Train Score', 'Test Score'], index=depths)

## Loop through the max_depth values
for depth in depths:
  ## Create an instance of the model
  dec_tree = DecisionTreeRegressor(max_depth=depth, random_state = 42)
  ## Create a model pipeline
  dec_tree_pipe = make_pipeline(preprocessor, dec_tree)
  ## Fit the model
  dec_tree_pipe.fit(X_train, y_train)

  ## Obtain the predictions from the model
  train_pred = dec_tree_pipe.predict(X_train)
  test_pred = dec_tree_pipe.predict(X_test)

  ## Obtain the R2 scores for Train and Test
  train_r2score = r2_score(y_train, train_pred)
  test_r2score = r2_score(y_test, test_pred)

  ## Save the Train and Test R2 Score for this depth in the scores dataframe
  scores.loc[depth, 'Train Score'] = train_r2score
  scores.loc[depth, 'Test Score'] = test_r2score

##Visualize the max_depths to display which achieves the highest R2 score
plt.plot(depths, scores['Train Score'], label='Train Score')
plt.plot(depths, scores['Test Score'], label='Test Score')
plt.ylabel('R2 Score')
plt.xlabel('Max Depth')
plt.legend()
plt.show()

## Create a version on the scores dataframe
## sorted by highest Test Scores
sorted_scores = scores.sort_values(by='Test Score', ascending=False)
## Display the first (5) rows of the dataframe
sorted_scores.head()

## sort the dataframe by test scores and save the index (k) of the best score
best_depth = sorted_scores.index[0]
best_depth

Evaluate the performance of your model based on RMSE and r^2.

#Calculating RMSE
train_RMSE= np.sqrt(np.mean(np.abs(training_predictions - y_train)**2))
test_RMSE = np.sqrt(np.mean(np.abs(test_predictions - y_test)**2))

print(f'Model Training RMSE: {train_RMSE}')
print(f'Model Testing RMSE: {test_RMSE}')

#Calculating r2
train_r2= r2_score(y_train, training_predictions)
test_r2= r2_score(y_test, test_predictions)

print(f'Model Training r2: {train_r2}')
print(f'Model Testing r2: {test_r2}')

Radom Forest Model

## Create an instance of the model
ran_for = RandomForestRegressor(random_state = 42)
## Create a model pipeline
ran_for_pipe = make_pipeline(preprocessor, ran_for)
## Fit the model
ran_for_pipe.fit(X_train, y_train)

## Display model performance metrics using a function
model_metrics(ran_for_pipe, x_train=X_train, y_train=y_train, 
                          x_test=X_test, y_test=y_test, 
                           model_name='Random Forest Model')

Tune Model

## Display the list of available hyperparameters for tuning
ran_for.get_params()

## Obtain the depths from the model using the estimators_ method
est_depths = [estimator.get_depth() for estimator in ran_for.estimators_]
## Assign the max est_depths value to max_depth variable
max_depth = max(est_depths)
## Display max_depth
max_depth

## Plot the scores
plt.plot(scores['Test Score'])
plt.plot(scores['Train Score'])
plt.show()

## Create a version on the scores dataframe
## sorted by highest Test Scores
sorted_scores = scores.sort_values(by='Test Score', ascending=False)
## Display the first (5) rows of the dataframe
sorted_scores.head()

## sort the dataframe by test scores and save the index (k) of the best score
best_depth = sorted_scores.index[0]
## Display best_depth
best_depth

Evaluate the model

## Create an instance of the model
ran_for = RandomForestRegressor(max_depth=best_depth, random_state=42)
## Create a model pipeline
ran_for_pipe = make_pipeline(preprocessor, ran_for)
## Fit the model
ran_for_pipe.fit(X_train, y_train)

## Display model performance metrics using a function
model_metrics(ran_for_pipe, x_train=X_train, y_train=y_train, 
                          x_test=X_test, y_test=y_test, 
                           model_name='Tuned Random Forest Model')





1.  Decision Tree Model Test Scores

MAE: 1037.7232 

MSE: 2190951.7954 

RMSE: 1480.1864 

R2: 0.2058826



2.   Random Forest Model Test Scores

MAE: 765.3043 

MSE: 1215973.5293 

RMSE: 1102.7119 

R2: 0.5592666







