# # # import pandas as pd
# # # import numpy as np
# # # import torch
# # # import torch.nn as nn
# # # import torch.optim as optim
# # # from sklearn.model_selection import train_test_split
# # # from sklearn.preprocessing import StandardScaler, OneHotEncoder
# # # from sklearn.compose import ColumnTransformer
# # # from torch.utils.data import DataLoader, TensorDataset

# # # # Load the data
# # # train_data = pd.read_csv('train.csv')
# # # test_data = pd.read_csv('test.csv')

# # # # Ensure the 'patient_id' column exists
# # # assert 'patient_id' in test_data.columns, "The test data must contain a 'patient_id' column."

# # # # Preprocess the data
# # # categorical_cols = [cname for cname in train_data.columns if train_data[cname].dtype == "object"]
# # # numerical_cols = [cname for cname in train_data.columns if train_data[cname].dtype in ['int64', 'float64']]

# # # # Remove target column from numerical columns if present
# # # if 'metastatic_diagnosis_period' in numerical_cols:
# # #     numerical_cols.remove('metastatic_diagnosis_period')

# # # # Define the preprocessors
# # # numerical_transformer = StandardScaler()
# # # categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# # # # Combine preprocessors
# # # preprocessor = ColumnTransformer(
# # #     transformers=[
# # #         ('num', numerical_transformer, numerical_cols),
# # #         ('cat', categorical_transformer, categorical_cols)
# # #     ])

# # # # Separate features and target
# # # X = train_data.drop('metastatic_diagnosis_period', axis=1)
# # # y = train_data['metastatic_diagnosis_period']

# # # # Split the data into training and validation sets
# # # X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# # # # Apply preprocessing to the data
# # # X_train = preprocessor.fit_transform(X_train).astype(np.float32)
# # # X_valid = preprocessor.transform(X_valid).astype(np.float32)
# # # X_test = preprocessor.transform(test_data).astype(np.float32)
# # # y_train = y_train.values.astype(np.float32)
# # # y_valid = y_valid.values.astype(np.float32)

# # # # Create TensorDatasets
# # # train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
# # # valid_dataset = TensorDataset(torch.tensor(X_valid), torch.tensor(y_valid))

# # # # Create DataLoaders
# # # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# # # valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# # # # Define the neural network
# # # class Net(nn.Module):
# # #     def __init__(self):
# # #         super(Net, self).__init__()
# # #         self.fc1 = nn.Linear(X_train.shape[1], 128)
# # #         self.dropout1 = nn.Dropout(0.2)
# # #         self.fc2 = nn.Linear(128, 64)
# # #         self.dropout2 = nn.Dropout(0.2)
# # #         self.fc3 = nn.Linear(64, 32)
# # #         self.fc4 = nn.Linear(32, 1)

# # #     def forward(self, x):
# # #         x = torch.relu(self.fc1(x))
# # #         x = self.dropout1(x)
# # #         x = torch.relu(self.fc2(x))
# # #         x = self.dropout2(x)
# # #         x = torch.relu(self.fc3(x))
# # #         x = self.fc4(x)
# # #         return x

# # # # Initialize the model, loss function, and optimizer
# # # model = Net()
# # # criterion = nn.MSELoss()
# # # optimizer = optim.Adam(model.parameters(), lr=0.001)

# # # # Training the model
# # # num_epochs = 50
# # # for epoch in range(num_epochs):
# # #     model.train()
# # #     for inputs, targets in train_loader:
# # #         optimizer.zero_grad()
# # #         outputs = model(inputs)
# # #         loss = criterion(outputs, targets.view(-1, 1))
# # #         loss.backward()
# # #         optimizer.step()

# # #     model.eval()
# # #     valid_loss = 0
# # #     with torch.no_grad():
# # #         for inputs, targets in valid_loader:
# # #             outputs = model(inputs)
# # #             loss = criterion(outputs, targets.view(-1, 1))
# # #             valid_loss += loss.item()

# # #     print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {valid_loss/len(valid_loader):.4f}')

# # # # Predict on test data
# # # model.eval()
# # # with torch.no_grad():
# # #     test_tensor = torch.tensor(X_test)
# # #     predictions = model(test_tensor).numpy().flatten()

# # # # Output predictions
# # # test_data['predicted_metastatic_diagnosis_period'] = predictions

# # # # Verify predictions are added correctly
# # # # assert 'predicted_metastatic_diagnosis_period' in test_data.columns, "The predictions were not added to the DataFrame."
# # # # assert not test_data['predicted_metastatic_diagnosis_period'].isnull().any(), "There are missing values in the predictions."

# # # # Save only patient_id and predicted_metastatic_diagnosis_period columns
# # # output = test_data[['patient_id', 'predicted_metastatic_diagnosis_period']]
# # # output.to_csv('test_predictions_pytorch.csv', index=False)
# # # print("Predictions saved successfully to test_predictions_pytorch.csv.")

# # import pandas as pd
# # from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.ensemble import RandomForestRegressor
# # from sklearn.metrics import mean_squared_error
# # import eli5
# # from eli5.sklearn import PermutationImportance
# # import shap
# # import lime
# # import lime.lime_tabular
# # from yellowbrick.model_selection import FeatureImportances
# # from alibi.explainers import AnchorTabular
# # # import lucid -- Lucid might not be directly applicable here, but included for completeness

# # # Load the data
# # train_data = pd.read_csv('train.csv')
# # test_data = pd.read_csv('test.csv')

# # # Data exploration (example)
# # print(train_data.head())
# # print(train_data.info())
# # print(train_data.describe())

# # # Assuming 'metastatic_diagnosis_period' is the target and 'patient_id' is an identifier
# # X_train = train_data.drop(['metastatic_diagnosis_period', 'patient_id'], axis=1)
# # y_train = train_data['metastatic_diagnosis_period']
# # X_test = test_data.drop(['patient_id'], axis=1)
# # test_ids = test_data['patient_id']

# # # Preprocess the data (e.g., scaling numerical features)
# # scaler = StandardScaler()
# # X_train_scaled = scaler.fit_transform(X_train)
# # X_test_scaled = scaler.transform(X_test)

# # # Train a predictive model
# # model = RandomForestRegressor(n_estimators=100, random_state=42)
# # model.fit(X_train_scaled, y_train)

# # # Evaluate the model
# # train_predictions = model.predict(X_train_scaled)
# # print(f"Train RMSE: {mean_squared_error(y_train, train_predictions, squared=False)}")

# # # Explain the model using ELI5
# # perm = PermutationImportance(model, random_state=42).fit(X_train_scaled, y_train)
# # eli5.show_weights(perm, feature_names=X_train.columns.tolist())

# # # Explain the model using SHAP
# # explainer = shap.TreeExplainer(model)
# # shap_values = explainer.shap_values(X_train_scaled)
# # shap.summary_plot(shap_values, X_train, feature_names=X_train.columns)

# # # Explain the model using LIME
# # lime_explainer = lime.lime_tabular.LimeTabularExplainer(X_train_scaled, feature_names=X_train.columns, class_names=['metastatic_diagnosis_period'], mode='regression')
# # lime_exp = lime_explainer.explain_instance(X_train_scaled[0], model.predict)
# # lime_exp.show_in_notebook()

# # # Explain the model using Yellowbrick
# # viz = FeatureImportances(model, labels=X_train.columns)
# # viz.fit(X_train_scaled, y_train)
# # viz.show()

# # # Explain the model using Alibi
# # anchor_explainer = AnchorTabular(model.predict, feature_names=X_train.columns)
# # anchor_explainer.fit(X_train_scaled, disc_perc=(25, 50, 75))
# # anchor_exp = anchor_explainer.explain(X_train_scaled[0])
# # print(anchor_exp)

# # # Predict on the test data
# # test_predictions = model.predict(X_test_scaled)

# # # Save the results to a file
# # results = pd.DataFrame({'patient_id': test_ids, 'metastatic_diagnosis_period': test_predictions})
# # results.to_csv('predictions.csv', index=False)

# # print("Predictions saved to 'predictions.csv'")



# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

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

# # Evaluation
# y_val_pred_gbr = gbr.predict(X_val)
# mae_gbr = mean_absolute_error(y_val, y_val_pred_gbr)
# mse_gbr = mean_squared_error(y_val, y_val_pred_gbr)
# print(f'Gradient Boosting Regressor - MAE: {mae_gbr}, MSE: {mse_gbr}')

# # Neural Network using TensorFlow
# model = Sequential()
# model.add(Dense(64, input_dim=X_train_processed.shape[1], activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1, activation='linear'))

# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# # Training the model
# history = model.fit(X_train_part, y_train_part, validation_data=(X_val, y_val), epochs=60, batch_size=32, verbose=1)

# # Evaluation
# y_val_pred_nn = model.predict(X_val)
# mae_nn = mean_absolute_error(y_val, y_val_pred_nn)
# mse_nn = mean_squared_error(y_val, y_val_pred_nn)
# print(f'Neural Network - MAE: {mae_nn}, MSE: {mse_nn}')

# # Predict on test data using both models
# y_test_pred_gbr = gbr.predict(X_test_processed)
# y_test_pred_nn = model.predict(X_test_processed)

# # Save the predictions
# test_data['metastatic_diagnosis_period_pred_gbr'] = y_test_pred_gbr
# test_data['metastatic_diagnosis_period_pred_nn'] = y_test_pred_nn

# test_data.to_csv('test_data_with_pred.csv', index=False)



# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset

# # Convert data to PyTorch tensors
# X_train_tensor = torch.tensor(X_train_part, dtype=torch.float32)
# y_train_tensor = torch.tensor(y_train_part.values, dtype=torch.float32)
# X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
# y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)
# X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)

# # Define the neural network architecture using PyTorch
# class PyTorchModel(nn.Module):
#     def __init__(self, input_size):
#         super(PyTorchModel, self).__init__()
#         self.fc1 = nn.Linear(input_size, 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.fc3 = nn.Linear(32, 1)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# # Create an instance of the model
# pytorch_model = PyTorchModel(X_train_tensor.shape[1])

# # Define loss function and optimizer
# criterion = nn.MSELoss()
# optimizer = optim.Adam(pytorch_model.parameters(), lr=0.001)

# # Training the PyTorch model
# def train_model(model, criterion, optimizer, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
#     train_dataset = TensorDataset(X_train, y_train)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     for epoch in range(epochs):
#         model.train()
#         running_loss = 0.0
#         for inputs, labels in train_loader:
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels.unsqueeze(1))
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#         # Validation loss
#         model.eval()
#         with torch.no_grad():
#             val_outputs = model(X_val)
#             val_loss = criterion(val_outputs, y_val.unsqueeze(1))
#         print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss.item():.4f}")

# train_model(pytorch_model, criterion, optimizer, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor)

# # Prediction using PyTorch
# pytorch_model.eval()
# with torch.no_grad():
#     y_val_pred_pytorch = pytorch_model(X_val_tensor).numpy()
#     y_test_pred_pytorch = pytorch_model(X_test_tensor).numpy()

# # Evaluation
# mae_pytorch = mean_absolute_error(y_val, y_val_pred_pytorch)
# mse_pytorch = mean_squared_error(y_val, y_val_pred_pytorch)
# print(f'PyTorch Model - MAE: {mae_pytorch}, MSE: {mse_pytorch}')

# # Predict on test data using PyTorch
# test_data['metastatic_diagnosis_period_pred_pytorch'] = y_test_pred_pytorch

# test_data.to_csv('test_data_with_predicti.csv', index=False)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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

# Split the training data for validation
X_train_part, X_val, y_train_part, y_val = train_test_split(X_train_processed, y_train, test_size=0.2, random_state=42)

# Gradient Boosting Regressor
gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)
gbr.fit(X_train_part, y_train_part)

# Evaluation
y_val_pred_gbr = gbr.predict(X_val)
mae_gbr = mean_absolute_error(y_val, y_val_pred_gbr)
mse_gbr = mean_squared_error(y_val, y_val_pred_gbr)
print(f'Gradient Boosting Regressor - MAE: {mae_gbr}, MSE: {mse_gbr}')

# Neural Network using TensorFlow
model = Sequential()
model.add(Dense(64, input_dim=X_train_processed.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Training the model
history = model.fit(X_train_part, y_train_part, validation_data=(X_val, y_val), epochs=50, batch_size=32, verbose=1)

# Evaluation
y_val_pred_nn = model.predict(X_val)
mae_nn = mean_absolute_error(y_val, y_val_pred_nn)
mse_nn = mean_squared_error(y_val, y_val_pred_nn)
print(f'Neural Network - MAE: {mae_nn}, MSE: {mse_nn}')

# Predict on test data using both models
y_test_pred_gbr = gbr.predict(X_test_processed)
y_test_pred_nn = model.predict(X_test_processed)

# Save the predictions
test_data['metastatic_diagnosis_period_pred_gbr'] = y_test_pred_gbr
test_data['metastatic_diagnosis_period_pred_nn'] = y_test_pred_nn

test_data.to_csv('test_data_with_predictions.csv', index=False)



from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

# Gradient Boosting Regressor
gbr = GradientBoostingRegressor(random_state=42)

# Define hyperparameters for tuning
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

# Use GridSearchCV to find the best hyperparameters
grid_search_gbr = GridSearchCV(gbr, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_gbr.fit(X_train_part, y_train_part)

best_gbr = grid_search_gbr.best_estimator_

# Evaluation
y_val_pred_gbr = best_gbr.predict(X_val)
mae_gbr = mean_absolute_error(y_val, y_val_pred_gbr)
mse_gbr = mean_squared_error(y_val, y_val_pred_gbr)
r2_gbr = r2_score(y_val, y_val_pred_gbr)
print(f'Gradient Boosting Regressor - MAE: {mae_gbr}, MSE: {mse_gbr}, R^2: {r2_gbr}')

# Random Forest Regressor
rf = RandomForestRegressor(random_state=42)

# Define hyperparameters for tuning
param_grid_rf = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 4, 5, None]
}

# Use GridSearchCV to find the best hyperparameters
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_rf.fit(X_train_part, y_train_part)

best_rf = grid_search_rf.best_estimator_

# Evaluation
y_val_pred_rf = best_rf.predict(X_val)
mae_rf = mean_absolute_error(y_val, y_val_pred_rf)
mse_rf = mean_squared_error(y_val, y_val_pred_rf)
r2_rf = r2_score(y_val, y_val_pred_rf)
print(f'Random Forest Regressor - MAE: {mae_rf}, MSE: {mse_rf}, R^2: {r2_rf}')

# Neural Network using TensorFlow
model = Sequential()
model.add(Dense(64, input_dim=X_train_processed.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Define a learning rate scheduler
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)

# Training the model with early stopping and learning rate scheduler
history = model.fit(X_train_part, y_train_part, validation_data=(X_val, y_val), epochs=100, batch_size=32, 
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10), lr_scheduler], verbose=1)

# Evaluate the model
y_val_pred_nn = model.predict(X_val)
mae_nn = mean_absolute_error(y_val, y_val_pred_nn)
mse_nn = mean_squared_error(y_val, y_val_pred_nn)
r2_nn = r2_score(y_val, y_val_pred_nn)
print(f'Neural Network - MAE: {mae_nn}, MSE: {mse_nn}, R^2: {r2_nn}')

# Final Predictions
y_test_pred_gbr = best_gbr.predict(X_test_processed)
y_test_pred_rf = best_rf.predict(X_test_processed)
y_test_pred_nn = model.predict(X_test_processed)

# Save the predictions
test_data['metastatic_diagnosis_period_pred_gbr'] = y_test_pred_gbr
test_data['metastatic_diagnosis_period_pred_rf'] = y_test_pred_rf
test_data['metastatic_diagnosis_period_pred_nn'] = y_test_pred_nn

test_data.to_csv('test_dahbsd.csv', index=False)
