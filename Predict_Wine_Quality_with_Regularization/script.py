import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegressionCV
import warnings
warnings.filterwarnings('ignore')

# Set plotting style for better visualization
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Load and explore the data
df = pd.read_csv('wine_quality.csv')
print("Dataset shape:", df.shape)
print("Column names:", df.columns)
print("\nSample data:")
print(df.head())
print("\nClass distribution:")
print(df['quality'].value_counts())
print("\nDescriptive statistics:")
print(df.describe())

# Check for missing values
missing_values = df.isnull().sum()
if missing_values.any():
    print("\nMissing values:")
    print(missing_values[missing_values > 0])
else:
    print("\nNo missing values found.")

# Split into features and target
y = df['quality']
features = df.drop(columns=['quality'])

# 1. Data transformation with StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(features)

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=99)
print(f"\nTraining set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# Helper function for model evaluation
def evaluate_model(clf, X_train, X_test, y_train, y_test, model_name):
    """Evaluate model performance and print results"""
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    
    train_score = f1_score(y_train, y_train_pred)
    test_score = f1_score(y_test, y_test_pred)
    
    print(f'\nPerformance for {model_name}:')
    print(f'Training F1 score: {train_score*100:.2f}%')
    print(f'Testing F1 score: {test_score*100:.2f}%')
    
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred))
    
    return train_score, test_score

# 3. Fit a logistic regression classifier without regularization
print("\n" + "="*50)
print("LOGISTIC REGRESSION WITHOUT REGULARIZATION")
print("="*50)
clf_no_reg = LogisticRegression(penalty=None, max_iter=1000)
clf_no_reg.fit(X_train, y_train)

# 4. Plot the coefficients
predictors = features.columns
coefficients = clf_no_reg.coef_.ravel()
coef = pd.Series(coefficients, predictors).sort_values()

plt.figure(figsize=(12, 8))
ax = coef.plot(kind='bar', title='Coefficients (No Regularization)')
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.tight_layout()
plt.savefig('no_regularization_coefficients.png')
plt.show()
plt.close()

# 5. Training and test performance
train_score_no_reg, test_score_no_reg = evaluate_model(clf_no_reg, X_train, X_test, y_train, y_test, "No-regularization Logistic Regression")

# 6. Default Implementation (L2-regularized!)
print("\n" + "="*50)
print("LOGISTIC REGRESSION WITH DEFAULT L2 REGULARIZATION")
print("="*50)
clf_default = LogisticRegression(max_iter=1000)
clf_default.fit(X_train, y_train)

# 7. Ridge Scores
train_score_default, test_score_default = evaluate_model(clf_default, X_train, X_test, y_train, y_test, "Default Regularization Logistic Regression (L2-penalty)")

# 8. Coarse-grained hyperparameter tuning
print("\n" + "="*50)
print("COARSE-GRAINED HYPERPARAMETER TUNING")
print("="*50)
training_array = []
test_array = []
C_array = [0.0001, 0.001, 0.01, 0.1, 1]

for c in C_array:
    print(f"Testing C={c}")
    clf = LogisticRegression(C=c, max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    train_f1 = f1_score(y_train, y_pred_train)
    test_f1 = f1_score(y_test, y_pred_test)
    training_array.append(train_f1)
    test_array.append(test_f1)
    print(f"Training F1: {train_f1*100:.2f}%, Test F1: {test_f1*100:.2f}%")

# 9. Plot training and test scores as a function of C
plt.figure(figsize=(10, 6))
plt.plot(C_array, training_array, 'o-', label='Training')
plt.plot(C_array, test_array, 'o-', label='Test')
plt.xscale('log')
plt.xlabel('Regularization Coefficient (C)')
plt.ylabel('F1 Score')
plt.title('Regularization Impact on Performance')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('regularization_impact.png')
plt.show()
plt.close()

# Create a DataFrame to display the results more clearly
results_df = pd.DataFrame({
    'C Value': C_array,
    'Training F1': [f"{score*100:.2f}%" for score in training_array],
    'Test F1': [f"{score*100:.2f}%" for score in test_array]
})
print("\nRegularization Impact Results:")
print(results_df)

# 10. Making a parameter grid for GridSearchCV
print("\n" + "="*50)
print("FINE-GRAINED HYPERPARAMETER TUNING WITH GRIDSEARCHCV")
print("="*50)
C_array = np.logspace(-4, -1, 100)
tuning_C = {'C': C_array}

# 11. Implementing GridSearchCV with l2 penalty
gs = GridSearchCV(
    LogisticRegression(max_iter=1000), 
    param_grid=tuning_C, 
    scoring='f1', 
    cv=5,
    verbose=1,
    n_jobs=-1  # Use all available cores
)
gs.fit(X_train, y_train)

# 12. Optimal C value and the score corresponding to it
best_C = gs.best_params_['C']
print(f'\nOptimal C value is {best_C:.2E}, with a cross-validation F1 score of {gs.best_score_ * 100:.2f}%')

# Plot the GridSearchCV results
cv_results = pd.DataFrame(gs.cv_results_)
plt.figure(figsize=(12, 6))
plt.semilogx(cv_results['param_C'], cv_results['mean_test_score'], 'o-')
plt.axvline(x=best_C, color='r', linestyle='--', label=f'Best C: {best_C:.2E}')
plt.xlabel('C Parameter')
plt.ylabel('Mean F1 Score (Cross-Validation)')
plt.title('GridSearchCV Results for L2 Regularization')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('gridsearch_results.png')
plt.show()
plt.close()

# 13. Validating the "best classifier"
print("\n" + "="*50)
print("VALIDATING THE BEST L2-REGULARIZED MODEL")
print("="*50)
clf_best_ridge = LogisticRegression(C=best_C, max_iter=1000)
clf_best_ridge.fit(X_train, y_train)

# Plot the coefficients for the best ridge model
coefficients = clf_best_ridge.coef_.ravel()
coef = pd.Series(coefficients, predictors).sort_values()

plt.figure(figsize=(12, 8))
ax = coef.plot(kind='bar', title=f'Coefficients (Best L2 Regularization, C={best_C:.2E})')
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.tight_layout()
plt.savefig('best_l2_coefficients.png')
plt.show()
plt.close()

train_score_best, test_score_best = evaluate_model(
    clf_best_ridge, X_train, X_test, y_train, y_test, 
    f"Best L2-Regularized Model (C={best_C:.2E})"
)

# 14. Implement L1 hyperparameter tuning with LogisticRegressionCV
print("\n" + "="*50)
print("L1 REGULARIZATION FOR FEATURE SELECTION")
print("="*50)
C_array_l1 = np.logspace(-2, 2, 100)
clf_l1 = LogisticRegressionCV(
    Cs=C_array_l1, 
    cv=5, 
    penalty='l1', 
    scoring='f1', 
    solver='liblinear',
    max_iter=1000,
    verbose=1,
    n_jobs=-1
)
clf_l1.fit(X, y)

# 15. Optimal C value and corresponding coefficients
best_C_l1 = clf_l1.C_[0]
print(f'\nOptimal C value for L1 Lasso is {best_C_l1:.2E}')
print("L1-regularized coefficients:")
l1_coefs = pd.Series(clf_l1.coef_.ravel(), index=predictors)
print(l1_coefs)

# Count non-zero coefficients (selected features)
nonzero_coefs = np.sum(clf_l1.coef_ != 0)
print(f"\nNumber of features selected by L1 regularization: {nonzero_coefs} out of {len(predictors)}")
print("Selected features:")
selected_features = l1_coefs[l1_coefs != 0].index.tolist()
print(selected_features)

# 16. Plotting the tuned L1 coefficients
coefficients = clf_l1.coef_.ravel()
coef = pd.Series(coefficients, predictors).sort_values()

plt.figure(figsize=(12, 8))
ax = coef.plot(kind='bar', title=f'Coefficients for L1 Regularization (C={best_C_l1:.2E})')
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.tight_layout()
plt.savefig('l1_coefficients.png')
plt.show()
plt.close()

# Validate L1 model
y_pred_l1 = clf_l1.predict(X)
l1_f1 = f1_score(y, y_pred_l1)
print(f"\nOverall F1 score for L1-regularized model: {l1_f1*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y, y_pred_l1))

# Model comparison summary
print("\n" + "="*50)
print("MODEL COMPARISON SUMMARY")
print("="*50)

models = [
    "No Regularization", 
    "Default L2 (C=1.0)", 
    f"Best L2 (C={best_C:.2E})",
    f"L1 Lasso (C={best_C_l1:.2E})"
]

train_scores = [
    train_score_no_reg, 
    train_score_default, 
    train_score_best,
    l1_f1  # Using overall score since we trained on all data
]

test_scores = [
    test_score_no_reg, 
    test_score_default, 
    test_score_best,
    l1_f1  # Using overall score since we trained on all data
]

summary_df = pd.DataFrame({
    'Model': models,
    'Training F1': [f"{score*100:.2f}%" for score in train_scores],
    'Test F1': [f"{score*100:.2f}%" for score in test_scores],
})

print(summary_df)

# Feature importance visualization combining all models
plt.figure(figsize=(14, 10))

# Create a dataframe with all coefficient values
coef_df = pd.DataFrame({
    'Feature': predictors,
    'No Regularization': clf_no_reg.coef_.ravel(),
    'Default L2': clf_default.coef_.ravel(),
    f'Best L2 (C={best_C:.2E})': clf_best_ridge.coef_.ravel(),
    f'L1 (C={best_C_l1:.2E})': clf_l1.coef_.ravel()
})

# Melt the dataframe for easier plotting
melted_df = pd.melt(coef_df, id_vars=['Feature'], var_name='Model', value_name='Coefficient')

# Plot
sns.barplot(x='Feature', y='Coefficient', hue='Model', data=melted_df)
plt.xticks(rotation=90)
plt.title('Feature Importance Comparison Across Models')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance_comparison.png')
plt.show()
plt.close()

print("\nAnalysis complete! All models have been evaluated and compared.")