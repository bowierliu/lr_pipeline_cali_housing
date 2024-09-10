# Model Comparison Using Pipelines on California Housing Data

## Project Motivation

The goal of this project is to use pipelines to run different machine learning models one at a time and compare their performances. By utilizing a consistent workflow with pipelines, we can streamline model training, preprocessing, and evaluation to find the best-performing model for predicting housing prices based on the California Housing Dataset.

## Libraries Used

- **Pandas**: For data loading, manipulation, and preprocessing.
- **NumPy**: For numerical operations.
- **Seaborn**: For visualizations.
- **Matplotlib**: For plotting data.
- **Scikit-learn**: For implementing machine learning models and creating pipelines.
  - Linear Regression
  - PowerTransformer
  - IterativeImputer
  - PCA and KernelPCA
- **XGBoost**: For implementing the XGBoost Regressor.

## Data

This project uses the California Housing Data, which contains the following features:
- **Numerical Features**:
  - `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`, `median_house_value`
- **Categorical Feature**:
  - `ocean_proximity`

### Data Loading

The data is loaded from a CSV file using a custom `load` function that prints basic information about the dataset, such as:
- Data dimensions
- Variable types (object, integer, float)
- Presence of missing values
- Memory usage

### Data Cleaning and Preparation

- **Missing Values**: The `total_bedrooms` column contains missing values. These missing values were imputed using an **IterativeImputer** with a **LinearRegression** estimator.
- **Feature Dropping**: The `ocean_proximity` column was dropped, as it was not considered relevant for predicting the target variable, `median_house_value`.
  
The cleaned data was split into features (X) and the target (y), and then further divided into training and testing sets using an 80-20 split.

### Data Visualization

A heatmap was generated to visualize the correlation between numerical features, and a pairplot was created to explore relationships between various features. Key observations:
- **Longitude** and **latitude** are related.
- **Total number of rooms** is positively correlated with **total bedrooms** and **households**.

## Models

### 1. Linear Regression Model
A pipeline was created to apply a **PowerTransformer** for feature scaling followed by a **Linear Regression** model.
```python
pipe = Pipeline([('pt', PowerTransformer()), ('lr', LinearRegression())])
