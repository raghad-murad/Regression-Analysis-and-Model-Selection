# Machine Learning Assignment 2: Regression Analysis and Model Selection

This repository contains the implementation and report for **Assignment 2** in the Machine Learning course. The assignment focuses on analyzing regression models using a dataset of cars scraped from the YallaMotors website. The goal is to predict car prices (`price_usd`) using various linear and nonlinear regression techniques.

---

## 📚 Table of Contents

- [Machine Learning Assignment 2: Regression Analysis and Model Selection](#machine-learning-assignment-2-regression-analysis-and-model-selection)
  - [📚 Table of Contents](#-table-of-contents)
  - [🌟 Overview](#-overview)
  - [📊 Dataset](#-dataset)
  - [🛠️ Implementation Details](#️-implementation-details)
  - [📁 Files in the Repository](#-files-in-the-repository)
    - [Main Files](#main-files)
    - [Data Files](#data-files)
    - [Additional Files](#additional-files)
  - [🚀 How to Run the Project](#-how-to-run-the-project)
    - [Prerequisites](#prerequisites)
    - [Steps to Run](#steps-to-run)
  - [📊 Results and Visualizations](#-results-and-visualizations)
  - [🤝 Contributions](#-contributions)
  - [📧 Contact](#-contact)
- [Thank you for checking out this project! 🚀](#thank-you-for-checking-out-this-project-)

---

## 🌟 Overview

The objective of this assignment is to:
1. **Preprocess the Dataset**: Handle missing values, encode categorical features, and normalize/standardize numerical features.
2. **Build Regression Models**:
   - Implement linear regression models (closed-form solution and gradient descent).
   - Apply LASSO (L1 Regularization) and Ridge (L2 Regularization) techniques to control overfitting.
   - Explore nonlinear models such as Polynomial Regression and Support Vector Regression (SVR).
3. **Evaluate Model Performance**: Use metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²) to assess model accuracy.
4. **Hyperparameter Tuning**: Use Grid Search to find optimal hyperparameters for Ridge, LASSO, and SVR models.
5. **Visualize Results**: Generate plots to compare model predictions with actual values, feature importances, and error distributions.

---

## 📊 Dataset

The dataset used in this project is titled **"YallaMotors Car Dataset"** and includes information about cars scraped from the YallaMotors website. It contains approximately **6,309 rows** and **9 columns**, with details such as car brand, engine capacity, horsepower, and price.

- **Features**: Include numerical and categorical attributes like `engine_capacity`, `cylinder`, `horse_power`, `top_speed`, and `seats`.
- **Target Variable**: `price_usd` (the price of the car in USD).

---

## 🛠️ Implementation Details

The project is implemented using Python with the following libraries:
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Matplotlib & Seaborn**: For data visualization.
- **Scikit-Learn**: For building and evaluating regression models.

The implementation is divided into modular parts, each focusing on a specific task:

1. **Data Preprocessing**:
   - Handle missing values using strategies like mean imputation or row removal.
   - Encode categorical features using binary encoding.
   - Normalize/standardize numerical features.

2. **Model Building**:
   - **Linear Regression**:
     - Closed-form solution: Solve the system of linear equations to obtain model parameters.
     - Gradient Descent: Implement gradient descent manually without external libraries.
   - **Regularization Techniques**:
     - LASSO (L1 Regularization): Penalizes large coefficients and zeros out less relevant features.
     - Ridge (L2 Regularization): Reduces model complexity by penalizing large coefficients.
   - **Nonlinear Models**:
     - Polynomial Regression: Apply polynomial transformations to features with degrees ranging from 2 to 10.
     - Support Vector Regression (SVR): Use an RBF kernel with hyperparameter tuning.

3. **Model Evaluation**:
   - Use metrics such as MSE, MAE, and R² to evaluate model performance.
   - Compare the performance of different models.

4. **Hyperparameter Tuning**:
   - Grid Search is used to find optimal hyperparameters for Ridge, LASSO, and SVR models.

5. **Visualization**:
   - Plot feature importances.
   - Visualize error distributions.
   - Compare model predictions with actual values.

---

## 📁 Files in the Repository

The repository contains the following files:

### Main Files
- **`Raghad_Murad_1212214_Assignment2_code.py`**: The main Python script that orchestrates all tasks (data preprocessing, model building, evaluation, and visualization).
- **`Raghad_Murad_1212214_Assignment2_report.pdf`**: Detailed report explaining the dataset, methodology, results, and analysis.

### Data Files
- **`cars.csv`**: The original dataset containing car information.
- **`train.csv`**, **`validation.csv`**, **`test.csv`**: Split datasets for training, validation, and testing.

### Additional Files
- **`results_visualizations.png`**: Plots and charts generated during the analysis.
- **`README.md`**: This file.

---

## 🚀 How to Run the Project

### Prerequisites
- **Python**: Ensure Python is installed on your machine.
- **Libraries**: Install the required libraries using `pip`:
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn
  ```

### Steps to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/raghad-murad/MachineLearningAssignment2.git
   ```

2. **Navigate to the Directory**
   ```bash
   cd MachineLearningAssignment2
   ```

3. **Run the Main Script**
   ```bash
   python Raghad_Murad_1212214_Assignment2_code.py
   ```

4. **View Results**
   - Output CSV files will be generated in the directory.
   - Visualizations will be displayed in separate windows or saved as images.

---

## 📊 Results and Visualizations

The project generates various outputs, including:
- **CSV Files**: Summarized statistics and transformed datasets.
- **Visualizations**:
  - Feature importance plots.
  - Error distribution histograms.
  - Comparison of model predictions vs. actual values.

Example visualizations include:
- Scatter plots comparing predicted vs. actual values.
- Bar charts showing feature importances.
- Line plots illustrating error trends.

---

## 🤝 Contributions

If you'd like to contribute to this repository, feel free to:
1. Fork the repository.
2. Create a new branch for your changes.
3. Submit a pull request with a detailed explanation of your changes.

---

## 📧 Contact

If you have any questions or suggestions, feel free to reach out!

- **Email:** raghadmbuzia@gmail.com
- **LinkedIn:** [in/raghad-murad](http://linkedin.com/in/raghad-murad-02690433a)

---

# Thank you for checking out this project! 🚀