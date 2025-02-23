import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import joblib

# Expanded Dataset
data = {
    "Age": [25, 54, 60, 37, 50, 42, 65, 52, 39, 58, 48, 34, 67, 53, 41, 59, 38, 66, 40, 63],
    "Sex": [0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1],
    "Chest Pain Type": [3, 2, 1, 0, 2, 3, 2, 1, 0, 3, 1, 0, 2, 3, 1, 2, 0, 3, 0, 1],
    "Resting BP": [120, 140, 130, 110, 150, 125, 145, 135, 118, 140, 128, 100, 160, 140, 130, 150, 110, 155, 112, 148],
    "Serum Cholesterol": [240, 260, 200, 210, 230, 220, 300, 180, 190, 250, 210, 180, 290, 270, 200, 250, 185, 280, 195, 240],
    "Fasting Blood Sugar": [0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
    "Resting ECG Results": [1, 0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 0, 1, 0, 2, 1, 2, 1, 1, 1],
    "Max HR": [150, 130, 120, 170, 140, 160, 135, 148, 155, 120, 138, 178, 132, 142, 154, 130, 165, 120, 158, 125],
    "Exercise Induced Angina": [0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1],
    "Oldpeak": [1.5, 2.3, 1.8, 0.6, 2.5, 0.8, 3.0, 1.2, 0.5, 2.0, 0.9, 0.3, 2.8, 1.7, 1.0, 2.5, 0.7, 2.9, 0.4, 2.2],
    "Slope": [2, 3, 1, 1, 3, 1, 3, 2, 1, 2, 1, 2, 3, 2, 1, 2, 1, 3, 2, 2],
    "Vessels": [3, 1, 2, 0, 3, 1, 2, 1, 0, 3, 1, 0, 2, 1, 0, 3, 0, 3, 0, 2],
    "Thal": [2, 3, 3, 1, 2, 1, 3, 2, 1, 2, 2, 1, 3, 2, 1, 3, 1, 3, 1, 2],
    "Heart Disease": [0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Features (X) and Target (y)
X = df.drop("Heart Disease", axis=1)
y = df["Heart Disease"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the Decision Tree model
tree_clf = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
tree_clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = tree_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Decision Tree Accuracy:", accuracy)

# Cross-validation
scores = cross_val_score(tree_clf, X, y, cv=5, scoring='accuracy')
print("Cross-validation Accuracy Scores:", scores)
print("Mean Cross-validation Accuracy:", scores.mean())

# Feature Importance
feature_importance = pd.Series(tree_clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importance:\n", feature_importance)

# Compare with other models
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train, y_train)
rf_accuracy = accuracy_score(y_test, rf_clf.predict(X_test))
print("\nRandom Forest Accuracy:", rf_accuracy)

lr_clf = LogisticRegression(max_iter=1000, random_state=42)
lr_clf.fit(X_train, y_train)
lr_accuracy = accuracy_score(y_test, lr_clf.predict(X_test))
print("Logistic Regression Accuracy:", lr_accuracy)

# Hyperparameter Tuning
param_grid = {
    "max_depth": [2, 3, 4, 5],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
print("\nBest Parameters from Grid Search:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

# Visualize the decision tree
print("\nDecision Tree Structure:")
tree_rules = export_text(tree_clf, feature_names=list(X.columns))
print(tree_rules)

plt.figure(figsize=(15, 10))
plot_tree(tree_clf, feature_names=X.columns, class_names=["No Disease", "Disease"], filled=True)
plt.show()

# Save the model
joblib.dump(tree_clf, "decision_tree_model.pkl")
print("\nModel saved as 'decision_tree_model.pkl'")

#Step 1: Model Evaluation and Refinement
#1.1 Hyperparameter Tuning
#This will help find the best max_depth, min_samples_split, and min_samples_leaf for your model.

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# Define the model
dt = DecisionTreeClassifier()

# Create a parameter grid to search for optimal hyperparameters
param_grid = {
    'max_depth': [2, 3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}

# Perform Grid Search
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

#Step 2: Feature Engineering
#2.2 One-Hot Encoding of Categorical Features
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Identify categorical columns
categorical_cols = ['Chest Pain Type', 'Resting ECG Results']

# Define the transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_cols)
    ],
    remainder='passthrough'
)

# Create a pipeline with preprocessing and model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier())
])

# Train the model with preprocessing
pipeline.fit(X_train, y_train)

#Step 3: Evaluation Metrics
#3.1 Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# First, fit the model to the training data
dt.fit(X_train, y_train)

# Now, make predictions on the test set
y_pred = dt.predict(X_test) 

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()

#3.2 ROC Curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Get false positive rate, true positive rate, and thresholds
fpr, tpr, thresholds = roc_curve(y_test, dt.predict_proba(X_test)[:, 1])

# Calculate AUC
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

"""
#Step 4: Model Deployment
#Model Deployment makes a trained machine learning model accessible for real-world use,
# allowing it to make predictions on new data.
from flask import Flask, request, jsonify
import pickle

# Load the trained model
with open('heart_disease_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get input from user
    data = request.get_json(force=True)
    input_data = [list(data.values())]
    
    # Make prediction
    prediction = model.predict(input_data)
    
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
"""