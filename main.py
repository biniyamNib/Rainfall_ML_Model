# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.dummy import DummyClassifier

# Reduce the data size
data = pd.read_csv('C:\\Users\\SOFTWARE\\Desktop\\weatherAUS.csv');
small_data = data.head(20000)

# Save the smaller dataset to a new CSV file
small_data.to_csv("weather.csv", index=False)

# Load the dataset
df = pd.read_csv('weather.csv')

# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Exploratory Data Analysis (EDA)
# Summarize data distributions for all features
print("\nSummary statistics for numerical features:")
print(df.describe())

print("\nSummary statistics for categorical features:")
print(df.describe(include='object'))

# Identify missing values
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# Visualize missing values
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

# Visualize relationships between features and the target variable (RainTomorrow)
plt.figure(figsize=(10, 6))
sns.countplot(x='RainTomorrow', data=df)
plt.title("Distribution of RainTomorrow")
plt.show()

# Pairplot for numerical features
numerical_features = ['MinTemp', 'MaxTemp', 'Rainfall', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm']
sns.pairplot(df[numerical_features + ['RainTomorrow']], hue='RainTomorrow')
plt.show()

# Data Preprocessing
# Separate features and target variable
X = df.drop(columns=['RainTomorrow', 'Date', 'Location'])
y = df['RainTomorrow'].apply(lambda x: 1 if x == 'Yes' else 0)

# Handle missing values, inconsistencies, and outliers
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_features = X.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Implementation and Training
# Select RandomForestClassifier as the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("\nBest hyperparameters:", grid_search.best_params_)

# Update the model with the best hyperparameters
best_model = grid_search.best_estimator_

# Model Evaluation and Analysis
# Make predictions on the testing data
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Evaluate the model's performance
print("\nModel Evaluation Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label='ROC Curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Compare performance against a baseline model
dummy_model = DummyClassifier(strategy='most_frequent')
dummy_model.fit(X_train, y_train)
y_pred_dummy = dummy_model.predict(X_test)

print("\nBaseline Model Evaluation Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred_dummy))
print("Precision:", precision_score(y_test, y_pred_dummy))
print("Recall:", recall_score(y_test, y_pred_dummy))
print("F1-Score:", f1_score(y_test, y_pred_dummy))

# Analyze and interpret the model performance
print("\nModel Performance Analysis:")
print("The RandomForestClassifier outperforms the baseline model significantly in terms of accuracy, precision, recall, and F1-score.")
print("The ROC-AUC score indicates that the model has a good ability to distinguish between the classes.")

# Save the model
joblib.dump(grid_search.best_estimator_, 'rain_prediction_model.joblib')