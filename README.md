Project 1 - Final

Jessica Uribe

This week, you will finalize your sales prediction project. The goal of this is to help the retailer understand the properties of products and outlets that play crucial roles in predicting sales.

Your first task is to build a linear regression model to predict sales.
Build a linear regression model. Evaluate the performance of your model based on r^2. Evaluate the performance of your model based on rmse. 2. Your second task is to build a regression tree model to predict sales.

Build a simple regression tree model. Compare the performance of your model based on r^2. Compare the performance of your model based on rmse. 3. You now have tried 2 different models on your data set. You need to determine which model to implement.

Overall, which model do you recommend? Justify your recommendation. 4. To finalize this project, complete a README in your GitHub repository including:

An overview of the project 2 relevant insights from the data (supported with reporting quality visualizations) Summary of the model and its evaluation metrics Final recommendations Here is a template you can use for your readme if you would like. You can look at the raw readme file to copy it if you want.

Please note:

Do not include detailed technical processes or code snippets in your README. If readers want to know more technical details they should be able to easily find your notebook to learn more. Make sure your GitHub repository is organized and professional. Remember, this should be used to showcase your data science skills and abilities. train test split Data

Load Library and data


# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, \
OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn import set_config
set_config(display='diagram')
     
Load the Data


df = pd.read_csv('/content/drive/MyDrive/Coding Dojo/sales_predictions.csv')
df.head()
     
Item_Identifier	Item_Weight	Item_Fat_Content	Item_Visibility	Item_Type	Item_MRP	Outlet_Identifier	Outlet_Establishment_Year	Outlet_Size	Outlet_Location_Type	Outlet_Type	Item_Outlet_Sales
0	FDA15	9.30	Low Fat	0.016047	Dairy	249.8092	OUT049	1999	Medium	Tier 1	Supermarket Type1	3735.1380
1	DRC01	5.92	Regular	0.019278	Soft Drinks	48.2692	OUT018	2009	Medium	Tier 3	Supermarket Type2	443.4228
2	FDN15	17.50	Low Fat	0.016760	Meat	141.6180	OUT049	1999	Medium	Tier 1	Supermarket Type1	2097.2700
3	FDX07	19.20	Regular	0.000000	Fruits and Vegetables	182.0950	OUT010	1998	NaN	Tier 3	Grocery Store	732.3800
4	NCD19	8.93	Low Fat	0.000000	Household	53.8614	OUT013	1987	High	Tier 3	Supermarket Type1	994.7052

df.copy()
     
Item_Identifier	Item_Weight	Item_Fat_Content	Item_Visibility	Item_Type	Item_MRP	Outlet_Identifier	Outlet_Establishment_Year	Outlet_Size	Outlet_Location_Type	Outlet_Type	Item_Outlet_Sales
0	FDA15	9.300	Low Fat	0.016047	Dairy	249.8092	OUT049	1999	Medium	Tier 1	Supermarket Type1	3735.1380
1	DRC01	5.920	Regular	0.019278	Soft Drinks	48.2692	OUT018	2009	Medium	Tier 3	Supermarket Type2	443.4228
2	FDN15	17.500	Low Fat	0.016760	Meat	141.6180	OUT049	1999	Medium	Tier 1	Supermarket Type1	2097.2700
3	FDX07	19.200	Regular	0.000000	Fruits and Vegetables	182.0950	OUT010	1998	NaN	Tier 3	Grocery Store	732.3800
4	NCD19	8.930	Low Fat	0.000000	Household	53.8614	OUT013	1987	High	Tier 3	Supermarket Type1	994.7052
...	...	...	...	...	...	...	...	...	...	...	...	...
8518	FDF22	6.865	Low Fat	0.056783	Snack Foods	214.5218	OUT013	1987	High	Tier 3	Supermarket Type1	2778.3834
8519	FDS36	8.380	Regular	0.046982	Baking Goods	108.1570	OUT045	2002	NaN	Tier 2	Supermarket Type1	549.2850
8520	NCJ29	10.600	Low Fat	0.035186	Health and Hygiene	85.1224	OUT035	2004	Small	Tier 2	Supermarket Type1	1193.1136
8521	FDN46	7.210	Regular	0.145221	Snack Foods	103.1332	OUT018	2009	Medium	Tier 3	Supermarket Type2	1845.5976
8522	DRG01	14.800	Low Fat	0.044878	Soft Drinks	75.4670	OUT046	1997	Small	Tier 1	Supermarket Type1	765.6700
8523 rows × 12 columns


## Display the number of rows and columns for the dataframe
df.shape
print(f'There are {df.shape[0]} rows, and {df.shape[1]} columns.')
print(f'The rows represent {df.shape[0]} observations, and the columns represent {df.shape[1]-1} features and 1 target variable.')
     
There are 8523 rows, and 12 columns.
The rows represent 8523 observations, and the columns represent 11 features and 1 target variable.

## Display the column names and datatypes for each column
## Columns with mixed datatypes are identified as an object datatype
df.dtypes
     
Item_Identifier               object
Item_Weight                  float64
Item_Fat_Content              object
Item_Visibility              float64
Item_Type                     object
Item_MRP                     float64
Outlet_Identifier             object
Outlet_Establishment_Year      int64
Outlet_Size                   object
Outlet_Location_Type          object
Outlet_Type                   object
Item_Outlet_Sales            float64
dtype: object

## Display the column names, count of non-null values, and their datatypes
df.info()
     
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 8523 entries, 0 to 8522
Data columns (total 12 columns):
 #   Column                     Non-Null Count  Dtype  
---  ------                     --------------  -----  
 0   Item_Identifier            8523 non-null   object 
 1   Item_Weight                7060 non-null   float64
 2   Item_Fat_Content           8523 non-null   object 
 3   Item_Visibility            8523 non-null   float64
 4   Item_Type                  8523 non-null   object 
 5   Item_MRP                   8523 non-null   float64
 6   Outlet_Identifier          8523 non-null   object 
 7   Outlet_Establishment_Year  8523 non-null   int64  
 8   Outlet_Size                6113 non-null   object 
 9   Outlet_Location_Type       8523 non-null   object 
 10  Outlet_Type                8523 non-null   object 
 11  Item_Outlet_Sales          8523 non-null   float64
dtypes: float64(4), int64(1), object(7)
memory usage: 799.2+ KB
Missing Values


## Display the descriptive statistics for the numeric columns
df.describe(include="number") # or 'object'
     
Item_Weight	Item_Visibility	Item_MRP	Outlet_Establishment_Year	Item_Outlet_Sales
count	7060.000000	8523.000000	8523.000000	8523.000000	8523.000000
mean	12.857645	0.066132	140.992782	1997.831867	2181.288914
std	4.643456	0.051598	62.275067	8.371760	1706.499616
min	4.555000	0.000000	31.290000	1985.000000	33.290000
25%	8.773750	0.026989	93.826500	1987.000000	834.247400
50%	12.600000	0.053931	143.012800	1999.000000	1794.331000
75%	16.850000	0.094585	185.643700	2004.000000	3101.296400
max	21.350000	0.328391	266.888400	2009.000000	13086.964800

## Display the descriptive statistics for the non-numeric columns
df.describe(exclude="number") # or 'object'
     
Item_Identifier	Item_Fat_Content	Item_Type	Outlet_Identifier	Outlet_Size	Outlet_Location_Type	Outlet_Type
count	8523	8523	8523	8523	6113	8523	8523
unique	1559	5	16	10	3	3	4
top	FDW13	Low Fat	Fruits and Vegetables	OUT027	Medium	Tier 3	Supermarket Type1
freq	10	5089	1232	935	2793	3350	5577

## Display the descriptive statistics for the non-numeric columns
df.describe(include="number")
     
Item_Weight	Item_Visibility	Item_MRP	Outlet_Establishment_Year	Item_Outlet_Sales
count	7060.000000	8523.000000	8523.000000	8523.000000	8523.000000
mean	12.857645	0.066132	140.992782	1997.831867	2181.288914
std	4.643456	0.051598	62.275067	8.371760	1706.499616
min	4.555000	0.000000	31.290000	1985.000000	33.290000
25%	8.773750	0.026989	93.826500	1987.000000	834.247400
50%	12.600000	0.053931	143.012800	1999.000000	1794.331000
75%	16.850000	0.094585	185.643700	2004.000000	3101.296400
max	21.350000	0.328391	266.888400	2009.000000	13086.964800
No unsual valies noted


print(f'There are {df.duplicated().sum()} duplicate rows.')
     
There are 0 duplicate rows.

#Identify rows, columns and missing values
df.info()
     
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 8523 entries, 0 to 8522
Data columns (total 12 columns):
 #   Column                     Non-Null Count  Dtype  
---  ------                     --------------  -----  
 0   Item_Identifier            8523 non-null   object 
 1   Item_Weight                7060 non-null   float64
 2   Item_Fat_Content           8523 non-null   object 
 3   Item_Visibility            8523 non-null   float64
 4   Item_Type                  8523 non-null   object 
 5   Item_MRP                   8523 non-null   float64
 6   Outlet_Identifier          8523 non-null   object 
 7   Outlet_Establishment_Year  8523 non-null   int64  
 8   Outlet_Size                6113 non-null   object 
 9   Outlet_Location_Type       8523 non-null   object 
 10  Outlet_Type                8523 non-null   object 
 11  Item_Outlet_Sales          8523 non-null   float64
dtypes: float64(4), int64(1), object(7)
memory usage: 799.2+ KB

#Display descriptive s tatitistics for all colums
df.describe(include='all')
     
Item_Identifier	Item_Weight	Item_Fat_Content	Item_Visibility	Item_Type	Item_MRP	Outlet_Identifier	Outlet_Establishment_Year	Outlet_Size	Outlet_Location_Type	Outlet_Type	Item_Outlet_Sales
count	8523	7060.000000	8523	8523.000000	8523	8523.000000	8523	8523.000000	6113	8523	8523	8523.000000
unique	1559	NaN	5	NaN	16	NaN	10	NaN	3	3	4	NaN
top	FDW13	NaN	Low Fat	NaN	Fruits and Vegetables	NaN	OUT027	NaN	Medium	Tier 3	Supermarket Type1	NaN
freq	10	NaN	5089	NaN	1232	NaN	935	NaN	2793	3350	5577	NaN
mean	NaN	12.857645	NaN	0.066132	NaN	140.992782	NaN	1997.831867	NaN	NaN	NaN	2181.288914
std	NaN	4.643456	NaN	0.051598	NaN	62.275067	NaN	8.371760	NaN	NaN	NaN	1706.499616
min	NaN	4.555000	NaN	0.000000	NaN	31.290000	NaN	1985.000000	NaN	NaN	NaN	33.290000
25%	NaN	8.773750	NaN	0.026989	NaN	93.826500	NaN	1987.000000	NaN	NaN	NaN	834.247400
50%	NaN	12.600000	NaN	0.053931	NaN	143.012800	NaN	1999.000000	NaN	NaN	NaN	1794.331000
75%	NaN	16.850000	NaN	0.094585	NaN	185.643700	NaN	2004.000000	NaN	NaN	NaN	3101.296400
max	NaN	21.350000	NaN	0.328391	NaN	266.888400	NaN	2009.000000	NaN	NaN	NaN	13086.964800
Split Data (validation Split)


# split X and y, you are predicting price
target = 'Item_Outlet_Sales'

X = df.drop(columns=target).copy()
y = df[target].copy()
X.head()
     
Item_Identifier	Item_Weight	Item_Fat_Content	Item_Visibility	Item_Type	Item_MRP	Outlet_Identifier	Outlet_Establishment_Year	Outlet_Size	Outlet_Location_Type	Outlet_Type
0	FDA15	9.30	Low Fat	0.016047	Dairy	249.8092	OUT049	1999	Medium	Tier 1	Supermarket Type1
1	DRC01	5.92	Regular	0.019278	Soft Drinks	48.2692	OUT018	2009	Medium	Tier 3	Supermarket Type2
2	FDN15	17.50	Low Fat	0.016760	Meat	141.6180	OUT049	1999	Medium	Tier 1	Supermarket Type1
3	FDX07	19.20	Regular	0.000000	Fruits and Vegetables	182.0950	OUT010	1998	NaN	Tier 3	Grocery Store
4	NCD19	8.93	Low Fat	0.000000	Household	53.8614	OUT013	1987	High	Tier 3	Supermarket Type1

# Perfoming a train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)
X_train.shape
     
(6392, 11)

display(X_train.info())
X_train.head()
     
<class 'pandas.core.frame.DataFrame'>
Int64Index: 6392 entries, 4776 to 7270
Data columns (total 11 columns):
 #   Column                     Non-Null Count  Dtype  
---  ------                     --------------  -----  
 0   Item_Identifier            6392 non-null   object 
 1   Item_Weight                5285 non-null   float64
 2   Item_Fat_Content           6392 non-null   object 
 3   Item_Visibility            6392 non-null   float64
 4   Item_Type                  6392 non-null   object 
 5   Item_MRP                   6392 non-null   float64
 6   Outlet_Identifier          6392 non-null   object 
 7   Outlet_Establishment_Year  6392 non-null   int64  
 8   Outlet_Size                4580 non-null   object 
 9   Outlet_Location_Type       6392 non-null   object 
 10  Outlet_Type                6392 non-null   object 
dtypes: float64(3), int64(1), object(7)
memory usage: 599.2+ KB
None
Item_Identifier	Item_Weight	Item_Fat_Content	Item_Visibility	Item_Type	Item_MRP	Outlet_Identifier	Outlet_Establishment_Year	Outlet_Size	Outlet_Location_Type	Outlet_Type
4776	NCG06	16.350	Low Fat	0.029565	Household	256.4646	OUT018	2009	Medium	Tier 3	Supermarket Type2
7510	FDV57	15.250	Regular	0.000000	Snack Foods	179.7660	OUT018	2009	Medium	Tier 3	Supermarket Type2
5828	FDM27	12.350	Regular	0.158716	Meat	157.2946	OUT049	1999	Medium	Tier 1	Supermarket Type1
5327	FDG24	7.975	Low Fat	0.014628	Baking Goods	82.3250	OUT035	2004	Small	Tier 2	Supermarket Type1
4810	FDD05	19.350	Low Fat	0.016645	Frozen Foods	120.9098	OUT045	2002	NaN	Tier 2	Supermarket Type1

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
     
ColumnTransformer(transformers=[('pipeline-1',
                                 Pipeline(steps=[('simpleimputer',
                                                  SimpleImputer(strategy='most_frequent')),
                                                 ('onehotencoder',
                                                  OneHotEncoder(handle_unknown='ignore',
                                                                sparse=False))]),
                                 <sklearn.compose._column_transformer.make_column_selector object at 0x7f21a73cc520>),
                                ('pipeline-2',
                                 Pipeline(steps=[('simpleimputer',
                                                  SimpleImputer()),
                                                 ('standardscaler',
                                                  StandardScaler())]),
                                 <sklearn.compose._column_transformer.make_column_selector object at 0x7f21a73cc1f0>)])
Please rerun this cell to show the HTML repr or trust the notebook.
Linear Regression

Create to predict sales


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


     
array([2970., 3766., 2276., 1232., 2156., -122., 1594., 4382., 3690.,
       1598.])

predictions = reg_pipe.predict(X_test)
     

prediction_df = X_test.copy()
prediction_df['True Median Price'] = y_test
prediction_df['Predicted Median Price'] = predictions
prediction_df['Error'] = predictions - y_test
prediction_df.head()
     
Item_Identifier	Item_Weight	Item_Fat_Content	Item_Visibility	Item_Type	Item_MRP	Outlet_Identifier	Outlet_Establishment_Year	Outlet_Size	Outlet_Location_Type	Outlet_Type	True Median Price	Predicted Median Price	Error
7503	FDI28	14.300	Low Fat	0.026300	Frozen Foods	79.4302	OUT013	1987	High	Tier 3	Supermarket Type1	1743.0644	882.0	-861.0644
2957	NCM17	7.930	Low Fat	0.071136	Health and Hygiene	42.7086	OUT046	1997	Small	Tier 1	Supermarket Type1	356.8688	1032.0	675.1312
7031	FDC14	14.500	Regular	0.041313	Canned	42.0454	OUT049	1999	Medium	Tier 1	Supermarket Type1	377.5086	1254.0	876.4914
1084	DRC36	NaN	Regular	0.044767	Soft Drinks	173.7054	OUT027	1985	Medium	Tier 3	Supermarket Type3	5778.4782	3752.0	-2026.4782
856	FDS27	10.195	Regular	0.012456	Meat	197.5110	OUT035	2004	Small	Tier 2	Supermarket Type1	2356.9320	1818.0	-538.9320

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
     
ColumnTransformer(transformers=[('pipeline-1',
                                 Pipeline(steps=[('simpleimputer',
                                                  SimpleImputer(strategy='most_frequent')),
                                                 ('onehotencoder',
                                                  OneHotEncoder(handle_unknown='ignore',
                                                                sparse=False))]),
                                 <sklearn.compose._column_transformer.make_column_selector object at 0x7f21a741f6a0>),
                                ('pipeline-2',
                                 Pipeline(steps=[('simpleimputer',
                                                  SimpleImputer()),
                                                 ('standardscaler',
                                                  StandardScaler())]),
                                 <sklearn.compose._column_transformer.make_column_selector object at 0x7f21a741f7f0>)])
Please rerun this cell to show the HTML repr or trust the notebook.
Make predictions using the testing data.

Evaluate the performance of your model based on RMSE and r^2.


#Calculating RMSE
train_RMSE = np.sqrt(np.mean(np.abs(training_predictions - y_train)**2))
test_RMSE= np.sqrt(np.mean(np.abs(test_predictions- y_test)**2))
print(f'Model Training RMSE: {train_RMSE}')
print(f'Model Testing RMSE: {test_RMSE}')
     
Model Training RMSE: 986.1228730794688
Model Testing RMSE: 28783081018469.617

#Calculating r2
train_r2= r2_score(y_train, training_predictions)
test_r2 = r2_score(y_test, test_predictions)

print(f'Model Training r2: {train_r2}')
print(f'Model Testing r2: {test_r2}')
     
Model Training r2: 0.6714131185261065
Model Testing r2: -3.002800191858062e+20
DECISION TREE


dec_tree = DecisionTreeRegressor(random_state = 42)
     

dec_tree = DecisionTreeRegressor(random_state=42)
dec_tree_pipe = make_pipeline(preprocessor, dec_tree)

#Fit using training data
dec_tree_pipe.fit(X_train, y_train)
     
Pipeline(steps=[('columntransformer',
                 ColumnTransformer(transformers=[('pipeline-1',
                                                  Pipeline(steps=[('simpleimputer',
                                                                   SimpleImputer(strategy='most_frequent')),
                                                                  ('onehotencoder',
                                                                   OneHotEncoder(handle_unknown='ignore',
                                                                                 sparse=False))]),
                                                  <sklearn.compose._column_transformer.make_column_selector object at 0x7f21a741f6a0>),
                                                 ('pipeline-2',
                                                  Pipeline(steps=[('simpleimputer',
                                                                   SimpleImputer()),
                                                                  ('standardscaler',
                                                                   StandardScaler())]),
                                                  <sklearn.compose._column_transformer.make_column_selector object at 0x7f21a741f7f0>)])),
                ('decisiontreeregressor',
                 DecisionTreeRegressor(random_state=42))])
Please rerun this cell to show the HTML repr or trust the notebook.

#Predict Values for train and test
train_preds = dec_tree_pipe.predict(X_train)
test_preds = dec_tree_pipe.predict(X_test)
     

train_score = dec_tree_pipe.score(X_train, y_train)
test_score = dec_tree_pipe.score(X_test, y_test)
print(train_score)
print(test_score)
     
1.0
0.21705461124441516

train_score = dec_tree_pipe.score(X_train, y_train)
test_score = dec_tree_pipe.score(X_test, y_test)
print(train_score)
print(test_score)
     
1.0
0.21705461124441516
Tune Model


dec_tree.get_params()
     
{'ccp_alpha': 0.0,
 'criterion': 'squared_error',
 'max_depth': None,
 'max_features': None,
 'max_leaf_nodes': None,
 'min_impurity_decrease': 0.0,
 'min_samples_leaf': 1,
 'min_samples_split': 2,
 'min_weight_fraction_leaf': 0.0,
 'random_state': 42,
 'splitter': 'best'}

#Calculating RMSE
train_RMSE= np.sqrt(np.mean(np.abs(training_predictions - y_train)**2))
test_RMSE = np.sqrt(np.mean(np.abs(test_predictions - y_test)**2))

print(f'Model Training RMSE: {train_RMSE}')
print(f'Model Testing RMSE: {test_RMSE}')
     
Model Training RMSE: 986.1228730794688
Model Testing RMSE: 28783081018469.617

#Calculating r2
train_r2= r2_score(y_train, training_predictions)
test_r2= r2_score(y_test, test_predictions)

print(f'Model Training r2: {train_r2}')
print(f'Model Testing r2: {test_r2}')
     
Model Training r2: 0.6714131185261065
Model Testing r2: -3.002800191858062e+20
Model recommended: Random Forest

It had the lowest error scores for Decision Tree, Bagged Tree and Random Forest, and it had the hightest Random Forest on the Test dataset. Random Forest Model Test Scores

Linear Regression Model

r2: 28783081018469.617

RMSE: -3.002800191858062e+20

Decison Tree Model

r2: 28783081018469.617

RMSE: -3.002800191858062e+2
