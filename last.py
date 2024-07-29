# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# import xgboost as xgb
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Concatenate

# # Load the data
# train_data = pd.read_csv('train.csv')
# test_data = pd.read_csv('test.csv')

# # Separate features and target
# X_train = train_data.drop(columns=['metastatic_diagnosis_period'])
# y_train = train_data['metastatic_diagnosis_period']
# X_test = test_data

# # Identify categorical and numerical columns
# categorical_cols = X_train.select_dtypes(include=['object']).columns
# numeric_cols = X_train.select_dtypes(include=['number']).columns

# # Define preprocessing for numeric and categorical data
# numeric_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='mean')),
#     ('scaler', StandardScaler())])

# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='most_frequent')),
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# # Combine preprocessing steps
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numeric_cols),
#         ('cat', categorical_transformer, categorical_cols)])

# # Preprocess the data
# X_train_processed = preprocessor.fit_transform(X_train)
# X_test_processed = preprocessor.transform(X_test)

# # Split the training data for validation
# X_train_part, X_val, y_train_part, y_val = train_test_split(X_train_processed, y_train, test_size=0.2, random_state=42)

# # Gradient Boosting Regressor
# gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)
# gbr.fit(X_train_part, y_train_part)

# # XGBoost Regressor
# xgboost_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
#                 max_depth = 5, alpha = 10, n_estimators = 100)
# xgboost_reg.fit(X_train_part, y_train_part)

# # Neural Network using TensorFlow
# model = Sequential()
# model.add(Dense(64, input_dim=X_train_processed.shape[1], activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1, activation='linear'))

# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# # Training the model
# history = model.fit(X_train_part, y_train_part, validation_data=(X_val, y_val), epochs=100, batch_size=32, verbose=1)

# # Use the models to make predictions on validation data
# y_val_pred_gbr = gbr.predict(X_val)
# y_val_pred_xgb = xgboost_reg.predict(X_val)
# y_val_pred_nn = model.predict(X_val)

# # Concatenate the predictions as features for the meta-model
# X_val_stacked = np.column_stack((y_val_pred_gbr, y_val_pred_xgb, y_val_pred_nn))

# # Meta-model (Neural Network)
# meta_model = Sequential()
# meta_model.add(Dense(64, input_dim=X_val_stacked.shape[1], activation='relu'))
# meta_model.add(Dense(32, activation='relu'))
# meta_model.add(Dense(1, activation='linear'))

# meta_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# # Train the meta-model
# meta_model.fit(X_val_stacked, y_val, epochs=100, batch_size=32, verbose=1)

# # Use the models to make predictions on test data
# y_test_pred_gbr = gbr.predict(X_test_processed)
# y_test_pred_xgb = xgboost_reg.predict(X_test_processed)
# y_test_pred_nn = model.predict(X_test_processed)

# # Concatenate the predictions as features for the meta-model
# X_test_stacked = np.column_stack((y_test_pred_gbr, y_test_pred_xgb, y_test_pred_nn))

# # Make predictions using the meta-model
# y_test_pred_meta = meta_model.predict(X_test_stacked)

# # Save the predictions
# test_data['metastatic_diagnosis_period_pred_meta'] = y_test_pred_meta

# test_data.to_csv('te.csv', index=False)


# # Add necessary imports
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.feature_selection import SelectFromModel
# from sklearn.model_selection import GridSearchCV

# # Additional feature engineering
# poly = PolynomialFeatures(degree=2)
# X_train_poly = poly.fit_transform(X_train_processed)
# X_val_poly = poly.transform(X_val)
# X_test_poly = poly.transform(X_test_processed)

# # Feature selection for meta-model
# selector = SelectFromModel(estimator=GradientBoostingRegressor())
# X_train_selected = selector.fit_transform(X_train_poly, y_train)
# X_val_selected = selector.transform(X_val_poly)
# X_test_selected = selector.transform(X_test_poly)

# # Hyperparameter tuning for base models
# param_grid_gbr = {'n_estimators': [100, 200, 300], 'learning_rate': [0.05, 0.1, 0.2]}
# gbr = GridSearchCV(GradientBoostingRegressor(), param_grid_gbr, cv=5)
# gbr.fit(X_train_processed, y_train)

# param_grid_xgb = {'n_estimators': [100, 200, 300], 'learning_rate': [0.05, 0.1, 0.2]}
# xgb_reg = GridSearchCV(xgb.XGBRegressor(objective ='reg:squarederror'), param_grid_xgb, cv=5)
# xgb_reg.fit(X_train_processed, y_train)

# # Hyperparameter tuning for meta-model
# param_grid_meta = {'hidden_layer_sizes': [(64,), (64, 32), (32, 16)], 'alpha': [0.01, 0.1, 1.0]}
# meta_model = GridSearchCV(MLPRegressor(max_iter=1000), param_grid_meta, cv=5)
# meta_model.fit(X_val_selected, y_val)

# # Use the tuned models to make predictions on test data
# y_test_pred_gbr = gbr.predict(X_test_processed)
# y_test_pred_xgb = xgb_reg.predict(X_test_processed)
# y_test_pred_nn = model.predict(X_test_processed)

# # Additional feature engineering for meta-model
# X_test_stacked = np.column_stack((y_test_pred_gbr, y_test_pred_xgb, y_test_pred_nn))

# # Use the tuned meta-model to make predictions
# y_test_pred_meta = meta_model.predict(X_test_stacked)

# # Save the predictions
# test_data['metastatic_diagnosis_period_pred_meta_tuned'] = y_test_pred_meta
# test_data.to_csv('test_da.csv', index=False)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.base import clone
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.neural_network import MLPRegressor
from category_encoders import TargetEncoder

# Load the data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Separate features and target
X_train = train_data.drop(columns=['metastatic_diagnosis_period'])
y_train = train_data['metastatic_diagnosis_period']
X_test = test_data

# Identify categorical and numerical columns
categorical_cols = X_train.select_dtypes(include=['object']).columns
numeric_cols = X_train.select_dtypes(include=['number']).columns

# Define preprocessing for numeric and categorical data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)])

# Preprocess the data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Additional feature engineering (e.g., target encoding)
target_encoder = TargetEncoder()
X_train_encoded = target_encoder.fit_transform(X_train, y_train)
X_test_encoded = target_encoder.transform(X_test)

# Split the training data for validation
X_train_part, X_val, y_train_part, y_val = train_test_split(X_train_encoded, y_train, test_size=0.2, random_state=42)

# Define base models
gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)
xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1, max_depth=5, alpha=10, n_estimators=100)
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Define and compile the neural network model
nn_model = Sequential([
    Dense(64, input_dim=X_train_encoded.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])
nn_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the neural network model
nn_model.fit(X_train_part, y_train_part, validation_data=(X_val, y_val), epochs=100, batch_size=32, verbose=1)

# Ensemble stacking with cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
meta_train = np.zeros((X_train_encoded.shape[0], 4))
meta_test = np.zeros((X_test_encoded.shape[0], 5, 4))

for fold, (train_index, val_index) in enumerate(kf.split(X_train_encoded)):
    X_train_fold, X_val_fold = X_train_encoded.iloc[train_index], X_train_encoded.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    # Clone and fit models on the fold
    gbr_fold = clone(gbr).fit(X_train_fold, y_train_fold)
    xgb_fold = clone(xgb_reg).fit(X_train_fold, y_train_fold)
    rf_fold = clone(rf).fit(X_train_fold, y_train_fold)
    nn_fold = clone(nn_model)
    nn_fold.fit(X_train_fold, y_train_fold, epochs=100, batch_size=32, verbose=0)

    # Predictions on validation fold
    meta_train[val_index, 0] = gbr_fold.predict(X_val_fold)
    meta_train[val_index, 1] = xgb_fold.predict(X_val_fold)
    meta_train[val_index, 2] = nn_fold.predict(X_val_fold).ravel()
    meta_train[val_index, 3] = rf_fold.predict(X_val_fold)

    # Predictions on test data for each fold
    meta_test[:, fold, 0] = gbr_fold.predict(X_test_encoded)
    meta_test[:, fold, 1] = xgb_fold.predict(X_test_encoded)
    meta_test[:, fold, 2] = nn_fold.predict(X_test_encoded).ravel()
    meta_test[:, fold, 3] = rf_fold.predict(X_test_encoded)

# Fit second-level meta-model on meta-training data
meta_model = MLPRegressor(hidden_layer_sizes=(64,), alpha=0.1, max_iter=1000)
meta_model.fit(meta_train, y_train)

# Ensemble predictions from second-level meta-model
y_test_pred_meta = meta_model.predict(meta_test.mean(axis=1))

# Save the predictions
test_data['metastatic_diagnosis_period_pred_meta'] = y_test_pred_meta
test_data.to_csv('test_data_wi.csv', index=False)

print('Meta-model predictions saved successfully.')
