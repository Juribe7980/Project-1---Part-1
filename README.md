Linear Regression

1-Create a modeling pipeline

[51]
0s
reg = LinearRegression()
Train the model

[52]
7s
reg_pipe = make_pipeline(preprocessor, reg)

reg_pipe.fit(X_train, y_train)

train_predictions = reg_pipe.predict(X_train)
test_predictions = reg_pipe.predict(X_test)
train_predictions[:10]
array([2992., 3832., 2320., 1248., 2240., -192., 1720., 4368., 3824.,
       1664.])
Make predictions using the testing data

[53]
0s
predictions = reg_pipe.predict(X_test)
[54]
1s
prediction_df = X_test.copy()
prediction_df['True Median Price'] = y_test
prediction_df['Predicted Median Price'] = predictions
prediction_df['Error'] = predictions - y_test
prediction_df.head()

Evaluate the performance of your model based on RMSE and r^2.

[55]
0s
#Calculating RMSE
train_RMSE = np.sqrt(np.mean(np.abs(train_predictions - y_train)**2))
test_RMSE= np.sqrt(np.mean(np.abs(test_predictions - y_test)**2))
print(f'Model Training RMSE: {train_RMSE}')
print(f'Model Testing RMSE: {test_RMSE}')
Model Training RMSE: 987.6414668668164
Model Testing RMSE: 15876039092305.197
[57]
0s
#Calculating r2
train_r2= r2_score(y_train, train_predictions)
test_r2 = r2_score(y_test, test_predictions)

print(f'Model Training r2: {train_r2}')
print(f'Model Testing r2: {test_r2}')
Model Training r2: 0.6704003153070058
Model Testing r2: -9.135581448036411e+19
the model shows a high error between the Model Test RMSE and the Model Training RMSE.

DECISION TREE MODEL***

[58]
0s
from sklearn.tree import DecisionTreeRegressor
dec_tree = DecisionTreeRegressor(random_state = 42)
[59]
1s
dec_tree = DecisionTreeRegressor(random_state=42)
dec_tree_pipe = make_pipeline(preprocessor, dec_tree)

#Fit using training data
dec_tree_pipe.fit(X_train, y_train)

[60]
0s
#Predict Values for train and test
train_preds = dec_tree_pipe.predict(X_train)
test_preds = dec_tree_pipe.predict(X_test)
[61]
0s
train_score = dec_tree_pipe.score(X_train, y_train)
test_score = dec_tree_pipe.score(X_test, y_test)
print(train_score)
print(test_score)
1.0
0.23769742878875222
Tune the Model

[62]
0s
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
[63]
0s
#Calculating RMSE
train_RMSE= np.sqrt(np.mean(np.abs(train_predictions - y_train)**2))
test_RMSE = np.sqrt(np.mean(np.abs(test_predictions - y_test)**2))

print(f'Model Training RMSE: {train_RMSE}')
print(f'Model Testing RMSE: {test_RMSE}')
Model Training RMSE: 987.6414668668164
Model Testing RMSE: 15876039092305.197
[ ]
0s
#Calculating r2
train_r2= np.corrcoef(y_train, train_predictions)[0][1]**2
test_r2 = np.corrcoef(y_test, test_predictions)[0][1]**2

print(f'Model Training r2: {train_r2}')
print(f'Model Testing r2: {test_r2}')
Bagged Trees

[64]
4s
from sklearn.ensemble import BaggingRegressor

## Create an instance of the model
bag_tree = BaggingRegressor(random_state = 42)
## Create a model pipeline
bag_tree_pipe = make_pipeline(preprocessor, bag_tree)
## Fit the model
bag_tree_pipe.fit(X_train, y_train)

[65]
0s
## Display the list of available hyperparameters for tuning
bag_tree.get_params()
{'base_estimator': None,
 'bootstrap': True,
 'bootstrap_features': False,
 'max_features': 1.0,
 'max_samples': 1.0,
 'n_estimators': 10,
 'n_jobs': None,
 'oob_score': False,
 'random_state': 42,
 'verbose': 0,
 'warm_start': False}
[66]
## List of estimator values
estimators = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

# Create a Dataframe to store the scores
scores = pd.DataFrame(index=estimators, columns=['Train Score', 'Test Score'])

# Loop through the values to find the best number of estimators
for num_estimators in estimators:
   # Create an instance of the model
   bag_reg = BaggingRegressor(n_estimators=num_estimators, random_state=42)
   # Create a model pipeline
   bag_reg_pipe = make_pipeline(preprocessor, bag_reg)
   # Fit the model
   bag_reg_pipe.fit(X_train, y_train)

   # Obtain the predictions from the model
   train_pred = bag_reg_pipe.predict(X_train)
   test_pred = bag_reg_pipe.predict(X_test)

   # Obtain the Train and Test R2 Scores
   train_r2score = r2_score(y_train, train_pred)
   test_r2score = r2_score(y_test, test_pred)

   # Save the Train and Test R2 Score for this num_estimators in the scores dataframe
   scores.loc[num_estimators, 'Train Score'] = train_r2score
   scores.loc[num_estimators, 'Test Score'] = test_r2score
[ ]
# Plot the scores
plt.plot(scores['Test Score'])
plt.plot(scores['Train Score'])
plt.show()
[ ]
## Create a version on the scores dataframe
## sorted by highest Test Scores
sorted_scores = scores.sort_values(by='Test Score', ascending=False)
## Display the first (5) rows of the dataframe
sorted_scores.head()
[ ]
## sort the dataframe by test scores and save the index (k) of the best score
best_depth = sorted_scores.index[0]
best_depth
[ ]
## Create an instance of the model
dec_tree = DecisionTreeRegressor(max_depth=best_depth, random_state = 42)
## Create a model pipeline
dec_tree_pipe = make_pipeline(preprocessor, dec_tree)
## Fit the model
dec_tree_pipe.fit(X_train, y_train)
[ ]
## Display the list of available hyperparameters for tuning
dec_tree.get_params()
Colab paid products - Cancel contracts here
arrow_right
