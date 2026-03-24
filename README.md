# Used Car Price Prediction with Decision Tree Regressor

## Overview

This project explores a dataset of used cars through exploratory data analysis (EDA) and builds a machine learning model to predict car selling prices using a **Decision Tree Regressor**.

The repository includes:

* The original dataset
* Jupyter notebooks for EDA and modeling
* A saved trained pipeline (`.pkl`) for prediction
* Python code for model training and inference

The main objective of this project is to identify a suitable model for predicting used car prices given both **numerical and categorical features**.

---

## Dataset

The dataset is located in the `datasets/` folder and contains car listings originally sourced from CarDekho.
Original dataset source: *https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho?select=car+data.csv*

### Features:

* **name** – Car model name
* **year** – Year of purchase
* **selling_price** – Listed selling price (target variable)
* **km_driven** – Total distance driven (in kilometers)
* **fuel** – Fuel type (Petrol, Diesel, etc.)
* **seller_type** – Type of seller (Individual, Dealer, etc.)
* **transmission** – Transmission type (Manual/Automatic)
* **owner** – Ownership history (First, Second, etc.)

---

## Exploratory Data Analysis (EDA)

EDA was performed using **pandas**, **matplotlib**, and **seaborn**.

Key observations:

* The dataset contains **outliers**, which were retained to preserve real-world data characteristics.
* Duplicate entries were identified and removed after data standardization.
* Feature relationships were analyzed using:

  * **Regression plots**
  * **Correlation heatmaps**

### Insights:

* `km_driven` and `car_age` show a **positive relationship** with each other.
* Several features show **negative correlation** with selling price.
* Categorical features (e.g., brand, fuel type) appear to significantly influence price.

---

## Model Development

### 1. Linear Regression (Baseline)

A Linear Regression model was initially tested (see `model_ver_1.ipynb`).

Findings:

* The model showed a generally good fit with low error.
* However, prediction errors increased for higher price values.
* The model struggled to fully capture the influence of **categorical variables**.

### Conclusion:

Linear Regression was not ideal due to the dataset’s mix of **categorical and numerical features**, and its assumption of linear relationships.

---

### 2. Decision Tree Regressor

A **Decision Tree Regressor** was selected as it:

* Handles both categorical (after encoding) and numerical data
* Captures non-linear relationships
* Provides interpretable feature importance

A pipeline was implemented using:

* `ColumnTransformer` for preprocessing
* `OneHotEncoder` for categorical variables
* `DecisionTreeRegressor` for modeling

---

## Model Evaluation

### Cross-Validation

K-Fold Cross Validation (k=5) was used to evaluate model performance.

* Initial testing with `max_depth=4` showed variability in performance
* Further tuning identified **max_depth=8** as optimal

### Results:

* Average MSE: ~0.19
* Standard Deviation: ~0.01

This indicates:

* Stable performance across folds
* Reduced risk of overfitting

---

## Feature Importance

Feature importance analysis from the Decision Tree model revealed:

* **car_age** has the strongest impact on price
* Followed by:

  * Fuel type
  * Transmission type
  * Brand

This aligns with real-world expectations in the used car market.

---

## Prediction Pipeline

The final model is saved as a `.pkl` file and can be used to predict prices for new user input.

The pipeline:

1. Preprocesses raw input (encoding + transformations)
2. Applies the trained Decision Tree model
3. Outputs predicted **log price**, which can be converted back using `exp()`

---

## Project Status Completed ✅

---

## Future Improvements

* Refactor code into reusable functions (EDA, preprocessing, modeling)
* Implement **GridSearchCV** for systematic hyperparameter tuning
* Explore other models (e.g., Random Forest, Gradient Boosting) for comparison

---

## Key Takeaways

* Model selection should align with **data characteristics**, not just performance
* Decision Trees are effective for datasets with mixed feature types
* Pipelines are essential for maintaining consistency between training and prediction

---

## Reflection
This project was a valuable learning experience where I was able to practice model selection, preprocessing, and evaluation techniques. Working through different approaches helped me better understand how to match a model to the nature of the dataset, especially when handling both categorical and numerical features.

I also gained more confidence in assessing model performance using cross-validation and interpreting feature importance.

While I am satisfied with the results, I recognize that there are still areas for improvement, particularly in optimizing hyperparameters and structuring the workflow more efficiently. These are aspects I aim to refine in future data science projects.

---
