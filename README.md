# Classify Raisins with Hyperparameter Tuning

This project demonstrates machine learning classification approaches to distinguish between different types of raisins using hyperparameter tuning techniques. The project utilizes Grid Search and Random Search methods to optimize model performance across multiple algorithms.

## Project Overview

The goal is to classify raisins into different categories based on their physical attributes. The project explores:

- Data exploration and visualization
- Model training with hyperparameter tuning
- Performance evaluation and comparison
- Decision boundary visualization

## Dataset

The project uses the Raisin Dataset, which contains physical measurements of raisins, including:
- Area
- Perimeter
- Major and minor axis lengths
- Eccentricity
- Convex area
- Extent

The target variable 'Class' represents different types of raisins.

## Methodology

Three classification algorithms were implemented and optimized:

1. **Decision Tree Classifier** with Grid Search CV
   - Hyperparameters tuned: max_depth, min_samples_split

2. **Logistic Regression** with Random Search CV
   - Hyperparameters tuned: penalty (L1/L2), regularization strength (C)

3. **K-Nearest Neighbors** with Grid Search CV
   - Hyperparameters tuned: n_neighbors, weights, distance metric

## Results

The models were evaluated using multiple metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrices

Visualizations include:
- Feature distributions
- Model accuracy comparisons
- Decision boundaries
- Confusion matrices

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- scikit-learn
- scipy

## How to Run

1. Ensure all dependencies are installed
2. Load the Raisin Dataset
3. Run the Jupyter notebook to execute the analysis pipeline

## File Structure

- `Classify_Raisins_with_Hyperparameter_Tuning.ipynb`: Main project notebook
- `Raisin_Dataset.csv`: Dataset containing raisin measurements
- `README.md`: Project documentation

## Conclusion

This project demonstrates the importance of hyperparameter tuning in optimizing machine learning models for classification tasks. The comparison of different algorithms provides insights into which approach works best for this particular dataset.