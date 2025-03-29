'''
Machine Learning Second Assignment
Assignment Title: Regression Analysis and Model Selection
Assignment Description:
    This assignment analyzes regression models using a YallaMotors car dataset to predict prices. We preprocessed the data, 
    then evaluated linear (e.g., linear regression, LASSO, Ridge) and nonlinear models (e.g., polynomial regression, Radial Basis Function kernel). 
    The SVR model with an RBF kernel performed best, achieving low MSE and high R-squared. Visualizations supported these findings. 
    Limitations include computational constraints and the need for more diverse data sources.
Student Information:
    - Name: Raghad Murad Buzia
    - ID: 1212214
    - Section: 3
'''

########################################################################################################################################
#                                                           0. About Dataset                                                           #
########################################################################################################################################

'''

Dataset Description:
This dataset contains information about cars scraped from the YallaMotors website.
It includes approximately 6,309 rows and 9 columns, providing details about various car features
such as engine capacity, cylinder power, horse power, top speed, and seating capacity.
The main objective of the dataset is to predict car prices based on these features.

---------------------------------------------------------------------------
|      Columns      |                     Description                     |
---------------------------------------------------------------------------
| - Car Name        | The name of the car (e.g., Toyota Corolla, BMW X5). |
| - Price           | The price of the car (may vary in currency).        |
| - Engine Capacity | The engine capacity of the car (e.g., 1.8L, 3.5L).  |
| - Cylinder        | The number of cylinders in the car engine.          |
| - Horse Power     | The horse power of the car.                         |
| - Top Speed       | The maximum speed of the car.                       |
| - Seats           | The number of seats in the car.                     |
| - Brand           | The car's brand (e.g., Toyota, BMW).                |
| - Country         | The country where the car is sold.                  |
---------------------------------------------------------------------------

'''

########################################################################################################################################
#                                                      1. Import Import Libraries                                                      #
########################################################################################################################################

'''
--> These lines import the necessary libraries for data analysis and visualization:
    - NumPy and pandas are used for data manipulation.
    - seaborn and matplotlib are for data visualization.
    - Scikit-learn libraries are imported for data preprocessing, including splitting datasets and encoding categorical variables.
'''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from category_encoders import BinaryEncoder
import re

########################################################################################################################################
#                                                         2. Data Preprocessing                                                        #
########################################################################################################################################

np.random.seed(0)                                                          # Set the random seed for reproducibility
warnings.filterwarnings('ignore')                                          # Ignore all warnings
sns.set_style("whitegrid", {'axes.grid' : False})                          # Set the seaborn style to whitegrid and disable grid on axes

# Load the dataset:
dataset = pd.read_csv('cars.csv')

# Data inspection:
print("\n- The number of rows and columns in the dataset is `{}` respectevly".format(dataset.shape))                     # Show the number of rows and columns in the dataset
print("\n- The first 5 rows of the dataset:\n{}".format(dataset.head()))                                                 # Display the first 5 rows of the dataset
print("\n- Summary of the dataset:\n")
print(dataset.info())                                                                                                    # Provide a concise summary of the dataset, including the number of non-null values and data types of each column
print("\n- Statistics descriptive for the dataset:\n{}".format(dataset.describe()))                                      # Generate descriptive statistics for the dataset, such as mean, standard deviation, min and max values for each numerical column
print("\n- The count of missing (NaN) values in each column of the dataset")
print(dataset.isna().sum())                                                                                              # Return the count of missing (NaN) values in each column of the dataset

##############################################    2.1 Cleaning and Handling Missing Data    ############################################

# List of known currencies
known_currencies = ['SAR', 'AED', 'USD', 'EGP', 'BHD', 'QAR', 'OMR', 'KWD']

# Function to identify valid currency or mark as None
def identify_currency_or_clean(price):
    # Check if the price contains a known currency
    for currency in known_currencies:
        if currency in str(price):
            return currency  # Return the currency
    # Return None if no valid currency is found
    return None

# Extract valid currencies
dataset['currency'] = dataset['price'].apply(identify_currency_or_clean)

# Display invalid price values (those with no known currency)
invalid_prices = dataset[dataset['currency'].isnull()]['price'].unique()

# Replace invalid prices with NaN
dataset.loc[dataset['currency'].isnull(), 'price'] = None

# Dictionary with exchange rates
exchange_rates = {
    'SAR': 0.27, 'AED': 0.27, 'USD': 1, 'EGP': 0.032, 'BHD': 2.65,
    'QAR': 0.27, 'OMR': 2.60, 'KWD': 3.30
}

# Function to convert prices to USD
def convert_to_usd(price, currency):
    if currency in exchange_rates:
        # Extract numeric value from the price and convert
        try:
            price_value = float(re.sub(r'[^\d.]', '', str(price)))  # Remove non-numeric characters
            return price_value * exchange_rates[currency]
        except ValueError:
            return None
    return None

# Apply the conversion to USD
dataset['price_usd'] = dataset.apply(lambda row: convert_to_usd(row['price'], row['currency']), axis=1)

# Fill missing values in the converted price column using the median
dataset['price_usd'].fillna(dataset['price_usd'].median(), inplace=True)

# Drop the original price and currency columns as they are no longer needed
dataset.drop(columns=['price', 'currency'], inplace=True)

# Convert other numeric-like columns to numeric
numeric_columns = ['engine_capacity', 'cylinder', 'horse_power', 'top_speed', 'seats']
for col in numeric_columns:
    dataset[col] = pd.to_numeric(dataset[col], errors='coerce')

# Fill missing values in numeric columns using the most frequent value
imputer = SimpleImputer(strategy='most_frequent')
dataset[numeric_columns] = imputer.fit_transform(dataset[numeric_columns])

################################################    2.2 Encoding Categorical Features    ###############################################

# Encode categorical features (brand, country, car name) using binary Encoding:
binary_encoder = BinaryEncoder(cols=['brand', 'country', 'car name'])
dataset = binary_encoder.fit_transform(dataset)

#####################################################    2.3 Normalizing/Standardizing    #################################################

# Normalize/Standardize numerical features
scaler = StandardScaler()
numeric_columns.append('price_usd')  # Include 'price' in the list of numeric columns to scale
dataset[numeric_columns] = scaler.fit_transform(dataset[numeric_columns])

#####################################################    2.4 Splitting the Dataset    ##################################################

# Split the data into training (60%), validation (20%), and testing (20%) sets:
train_data, temp_data = train_test_split(dataset, test_size=0.4, random_state=42)
validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Print the shapes of the datasets:
print("\n- Training data shape: the number of rows and columns in the dataset is `{}` respectevly".format(train_data.shape))
print("\n- Validation data shape: the number of rows and columns in the dataset is `{}` respectevly".format(validation_data.shape))
print("\n- Test data shape: the number of rows and columns in the dataset is `{}` respectevly\n".format(test_data.shape))

########################################################################################################################################
#                                                   3. Building Regression Models                                                      #
########################################################################################################################################

#########################################################    Evaluate models    ########################################################

# Function to evaluate models based on MSE, MAE, and R-squared
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, r2

#############################################    Feature Selection with Forward Selection    ###########################################

X_train = train_data.drop('price_usd', axis=1).values
y_train = train_data['price_usd'].values

X_val = validation_data.drop('price_usd', axis=1).values
y_val = validation_data['price_usd'].values

X_train_df = pd.DataFrame(X_train, columns=train_data.drop('price_usd', axis=1).columns)
X_val_df = pd.DataFrame(X_val, columns=train_data.drop('price_usd', axis=1).columns)

# Function to Forward Selection for Feature Selection
def forward_selection(X_train, y_train, X_val, y_val, max_features=None):
    selected_features = []
    remaining_features = list(X_train.columns)
    best_mse = float("inf")
    iteration = 0

    while remaining_features:
        mse_list = []
        for feature in remaining_features:
            model = LinearRegression()
            model.fit(X_train[selected_features + [feature]], y_train)
            y_val_pred = model.predict(X_val[selected_features + [feature]])
            mse = mean_squared_error(y_val, y_val_pred)
            mse_list.append((feature, mse))

        best_feature, best_mse_candidate = min(mse_list, key=lambda x: x[1])

        if best_mse_candidate < best_mse:
            best_mse = best_mse_candidate
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            iteration += 1
            print("Iteration {}: Adding feature '{}' with MSE: {:.4f}".format(iteration, best_feature, best_mse))

        elif max_features is not None and len(selected_features) >= max_features:
            print("Reached maximum number of features ({}). Stopping.".format(max_features))
            break
        else:
            print("No improvement in performance. Stopping.")
            break

    print("\nSelected features:", selected_features)
    return selected_features

# Run forward selection
selected_features = forward_selection(X_train_df, y_train, X_val_df, y_val, max_features=10)
X_train_selected = X_train_df[selected_features]
X_val_selected = X_val_df[selected_features]

############################################    3.1 Linear Regression: Closed-Form Solution    #########################################

# Closed-Form Solution
def closed_form_solution(X, y):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
    theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return theta

# Compute theta using Closed-Form Solution
theta_closed = closed_form_solution(X_train_selected, y_train)
print("\n- Theta from Closed-Form Solution: `{}`".format(theta_closed))

# Predictions using Closed-Form Solution
X_val_b = np.c_[np.ones((X_val.shape[0], 1)), X_val_selected]
y_val_pred_closed = X_val_b.dot(theta_closed)
mse_closed, mae_closed, r2_closed = evaluate_model(y_val, y_val_pred_closed)
print("\n- Closed-Form Solution - MSE: {:.4f}, MAE: {:.4f}, R2: {:.4f}".format(mse_closed, mae_closed, r2_closed))

##############################################    3.2 Linear Regression: Gradient Descent    ###########################################

# Gradient Descent Implementation
def gradient_descent(X, y, lr=0.01, epochs=1000):
    m = len(y)
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
    theta = np.random.randn(X_b.shape[1])  # Initialize theta randomly
    for _ in range(epochs):
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta -= lr * gradients
    return theta

theta_gd = gradient_descent(X_train_selected, y_train)
print("\n- Theta from Gradient Descent: `{}`".format(theta_gd))

# Predictions using Gradient Descent
y_val_pred_gd = np.c_[np.ones((X_val_selected.shape[0], 1)), X_val_selected].dot(theta_gd)
mse_gd, mae_gd, r2_gd = evaluate_model(y_val, y_val_pred_gd)
print("\n- Gradient Descent - MSE: {:.4f}, MAE: {:.4f}, R2: {:.4f}".format(mse_gd, mae_gd, r2_gd))

################################################    3.3 Lasso and Ridge Regularization    ##############################################

# Lasso Regression
lasso = Lasso(alpha=0.1)  # Regularization parameter
lasso.fit(X_train_selected, y_train)
y_val_pred_lasso = lasso.predict(X_val_selected)
mse_lasso, mae_lasso, r2_lasso = evaluate_model(y_val, y_val_pred_lasso)
print("\n- Lasso Regression - MSE: {:.4f}, MAE: {:.4f}, R2: {:.4f}".format(mse_lasso, mae_lasso, r2_lasso))

# Ridge Regression
ridge = Ridge(alpha=0.1)  # Regularization parameter
ridge.fit(X_train_selected, y_train)
y_val_pred_ridge = ridge.predict(X_val_selected)
mse_ridge, mae_ridge, r2_ridge = evaluate_model(y_val, y_val_pred_ridge)
print("\n- Ridge Regression - MSE: {:.4f}, MAE: {:.4f}, R2: {:.4f}".format(mse_ridge, mae_ridge, r2_ridge))

###########################################    3.4 Hyperparameter Tuning for Lasso and Ridge    ########################################

param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}

# Grid Search for Lasso
lasso_grid = GridSearchCV(Lasso(), param_grid, scoring='neg_mean_squared_error', cv=5)
lasso_grid.fit(X_train_selected, y_train)
best_lasso = lasso_grid.best_estimator_
print("\n- Best Lasso Alpha: `{}`".format(lasso_grid.best_params_))

# Grid Search for Ridge
ridge_grid = GridSearchCV(Ridge(), param_grid, scoring='neg_mean_squared_error', cv=5)
ridge_grid.fit(X_train_selected, y_train)
best_ridge = ridge_grid.best_estimator_
print("\n- Best Ridge Alpha: `{}`".format(ridge_grid.best_params_))

###################################################    3.5 Polynomial Regression    ####################################################

# Polynomial Regression with Degrees 2 to 10:

'''
for degree in range(2, 11):
    poly_features = PolynomialFeatures(degree=degree)
    X_poly_train = poly_features.fit_transform(X_train_selected)
    X_poly_val = poly_features.transform(X_val_selected)

    poly_model = LinearRegression()
    poly_model.fit(X_poly_train, y_train)
    y_val_pred_poly = poly_model.predict(X_poly_val)
    mse_poly, mae_poly, r2_poly = evaluate_model(y_val, y_val_pred_poly)
    print("\n- Polynomial Regression (Degree {}) - MSE: {:.4f}, MAE: {:.4f}, R2: {:.4f}".format(degree, mse_poly, mae_poly, r2_poly))

'''
# 1. Select a small sample of data (only 10%)
'''
(this is because the data is large and my laptop and the collab could not handle the full polynomial from 2 to 10)
'''
train_sample = train_data.sample(frac=0.1, random_state=42)

# 2. Select the features that were selected by Forward Selection
X_train_sample = train_sample[selected_features].values
y_train_sample = train_sample['price_usd'].values

# 3. Repeat across degrees 2 to 10 for Polynomial Regression
for degree in range(2, 9):
    print(f"\nProcessing Polynomial Regression with degree {degree}...")

    # 3.1. Applying Polynomial Features to the selected features
    poly_features = PolynomialFeatures(degree=degree)
    X_poly_train = poly_features.fit_transform(X_train_sample)

    # 3.2. Apply the same transformation to the validation set using the same selected features
    X_val_selected = validation_data[selected_features].values
    X_poly_val = poly_features.transform(X_val_selected)

    # 3.3. Train the Linear Regression model on the polynomial features
    poly_model = LinearRegression()
    poly_model.fit(X_poly_train, y_train_sample)

    # 3.4. Predicting values on the validation set and calculating MSE, MAE, and R2
    y_val_pred_poly = poly_model.predict(X_poly_val)
    mse_poly, mae_poly, r2_poly = evaluate_model(y_val, y_val_pred_poly)
    print("\n- Polynomial Regression (Degree {}) - MSE: {:.4f}, MAE: {:.4f}, R2: {:.4f}".format(degree, mse_poly, mae_poly, r2_poly))

##########################################    3.6 Support Vector Regression (RBF Kernel)    ############################################

# SVR with RBF Kernel
svr_model = SVR(kernel='rbf', C=100, gamma=0.1)
svr_model.fit(X_train_selected, y_train)
y_val_pred_svr = svr_model.predict(X_val_selected)
mse_svr, mae_svr, r2_svr = evaluate_model(y_val, y_val_pred_svr)
print("\n- Support Vector Regression (RBF Kernel) - MSE: {:.4f}, MAE: {:.4f}, R2: {:.4f}".format(mse_svr, mae_svr, r2_svr))

########################################################################################################################################
#                                            4. Choose The Best Models based MSE and R2                                                #
########################################################################################################################################

# Choose the best model based on the lowest MSE or highest R2
models = {
    "Closed-Form Solution": (mse_closed, r2_closed),
    "Gradient Descent": (mse_gd, r2_gd),
    "Lasso Regression": (mse_lasso, r2_lasso),
    "Ridge Regression": (mse_ridge, r2_ridge),
    "Polynomial Regression": (mse_poly, r2_poly),
    "SVR": (mse_svr, r2_svr)
}

# Determine the best model based on the lowest MSE or highest R2
best_model_by_mse = min(models, key=lambda model: models[model][0])
best_model_by_r2 = max(models, key=lambda model: models[model][1])
print("\nThe best models:")
print("\n- Best Model based on MSE: {}".format(best_model_by_mse))
print("\n- Best Model based on R2: {}".format(best_model_by_r2))

########################################################################################################################################
#                                             5. Hyperparameter Tuning with Grid Search                                                #
########################################################################################################################################

# Hyperparameter Tuning for SVR (the best model based both MSE and R):
svr_param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf']
}
svr_grid = GridSearchCV(SVR(), svr_param_grid, scoring='neg_mean_squared_error', cv=5)
svr_grid.fit(X_train_selected, y_train)
best_svr = svr_grid.best_estimator_
print("\n- Best SVR Parameters: {}".format(svr_grid.best_params_))

########################################################################################################################################
#                                                  6. Model Evaluation on Test Set                                                     #
########################################################################################################################################

# Prepare test data:
X_test = test_data[selected_features]  
y_test = test_data['price_usd'].values              

# Evaluate the best model (in our case it is SVR)
y_test_pred = best_svr.predict(X_test)  

# Performance metrics calculation
mse_test, mae_test, r2_test = evaluate_model(y_test, y_test_pred)
print("\n- Test Set Evaluation (SVR) - MSE: {:.4f}, MAE: {:.4f}, R2: {:.4f}\n\n".format(mse_test, mae_test, r2_test))

########################################################################################################################################
#                                         Visualizations for the Best Model (SVR)                                                     #
########################################################################################################################################

# 1. Error Distribution
def plot_error_distribution(y_true, y_pred, model_name="Model"):
    errors = y_true - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, bins=30, kde=True, color='purple', alpha=0.7)
    plt.title(f"{model_name} - Error Distribution")
    plt.xlabel("Prediction Errors")
    plt.ylabel("Frequency")
    plt.axvline(0, color='red', linestyle='--', label='Zero Error')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot Error Distribution for SVR
plot_error_distribution(y_test, y_test_pred, model_name="SVR")

print("\n\n")

# 2. Predictions vs Actual Values
def plot_predictions_vs_actual(y_true, y_pred, model_name="Model"):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, color='blue', label='Predicted vs Actual')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', label='Perfect Prediction')
    plt.title(f"{model_name} - Predictions vs Actual Values")
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot Predictions vs Actual Values for SVR
plot_predictions_vs_actual(y_test, y_test_pred, model_name="SVR")

print("\n\n")

# 3. Feature Importances (Permutation-based for SVR

def plot_svr_feature_importances(model, X, y, feature_names, model_name="Model"):
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42, scoring='neg_mean_squared_error')
    importances = result.importances_mean
    sorted_indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[sorted_indices], align='center', alpha=0.7, color='orange')
    plt.xticks(range(len(importances)), feature_names[sorted_indices], rotation=90)
    plt.title(f"{model_name} - Feature Importances (Permutation)")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()

# Plot Feature Importances for SVR using permutation importance
plot_svr_feature_importances(best_svr, X_test, y_test, X_test.columns, model_name="SVR")
