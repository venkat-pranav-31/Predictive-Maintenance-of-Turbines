# Predictive Maintenance for Wind Turbines

A machine learning project that predicts impending failures in wind turbines using sensor data, enabling proactive maintenance scheduling instead of reactive (and costly) breakdown repairs. The project goes beyond standard accuracy metrics by incorporating a **business-driven, cost-sensitive evaluation framework** to reflect the real financial impact of different prediction errors.

## Problem Statement

Wind turbines generate continuous sensor data (`V1`, `V2`, ... etc.) that can be used to detect early signs of mechanical failure. Given these sensor readings, the goal is to predict whether a turbine is likely to fail (`Target = 1`) or continue operating normally (`Target = 0`), allowing maintenance teams to act before a costly unplanned breakdown occurs.

## Why Cost-Sensitive Evaluation Matters

In predictive maintenance, not all prediction errors are equal:

| Outcome | Meaning | Estimated Cost |
|---------|---------|-----------------|
| True Positive | Failure correctly predicted | $15 (planned maintenance check) |
| False Positive | False alarm — turbine was actually fine | $5 (unnecessary check) |
| False Negative | Missed an actual failure | $40 (unplanned breakdown) |

A standard accuracy score would treat these errors equally, but in practice, **missing a real failure (False Negative) is far more expensive than a false alarm**. This project defines a custom cost function based on the confusion matrix and uses it — alongside standard metrics — to evaluate and tune the model, including testing different probability decision thresholds (0.5 vs. 0.42) to better balance this trade-off.

## Project Workflow

1. **Data Loading & Cleaning**
   - Load `Train.csv` and `Test.csv`
   - Identify and impute missing values (`V1`, `V2`) using mean imputation

2. **Train/Test Split** — 80/20 split of features and target for initial model development

3. **Custom Cost-Based Evaluation Functions**
   - `min_model_cost` / `min_model_cost1` — compute the ratio of theoretical minimum cost to actual cost incurred, at different decision thresholds (0.5 and 0.42)
   - `maintanence_cost` / `maintanence_cost1` — compute accuracy, precision, recall, F1-score, and the cost-based metric together
   - `confusion_matrix_sklearn` / `confusion_matrix_sklearn1` — visualize confusion matrices with both counts and percentages at each threshold

4. **Model: Artificial Neural Network (ANN)**
   - Built with Keras `Sequential` API
   - Architecture: Dense(256) → Dropout(0.5) → Dense(128) → Dropout(0.5) → Dense(64) → Dropout(0.5) → Dense(32) → Dropout(0.5) → Dense(1, sigmoid)
   - Compiled with Adam optimizer and binary cross-entropy loss

5. **Cross-Validation** — 5-fold `KFold` cross-validation to obtain a robust estimate of the cost-based score across different data splits

6. **Handling Class Imbalance**
   - **SMOTE (oversampling)** — synthetically generates additional minority-class (failure) samples
   - **Random Undersampling** — reduces majority-class (no-failure) samples
   - Both strategies are evaluated via cross-validation to compare their effect on the cost-based score

7. **Test Set Evaluation** — preprocessing and evaluation applied to a held-out test set, reporting both standard classification metrics and the custom maintenance-cost metric at multiple thresholds

8. **Final Production Pipeline** — combines `SimpleImputer` and the trained neural network into a single `sklearn` `Pipeline`, trained end-to-end on the full dataset and evaluated on the test set for deployment readiness

## Tech Stack

- **Language:** Python
- **Data Handling:** Pandas, NumPy
- **Modeling:** TensorFlow / Keras (Artificial Neural Network)
- **Imbalanced Data Handling:** imbalanced-learn (SMOTE, RandomUnderSampler)
- **Model Evaluation:** Scikit-learn (confusion matrix, classification report, cross-validation, custom scorers)
- **Visualization:** Matplotlib, Seaborn

## How to Run

1. Clone this repository and ensure `Train.csv` and `Test.csv` are placed in the project directory.
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow imbalanced-learn
   ```
3. Open `Predictive_maintenance_for_wind_turbine.ipynb` in Jupyter Notebook or Google Colab.
4. Run the cells sequentially:
   - Data preprocessing and imputation
   - ANN model training
   - Cross-validation and class-balancing experiments
   - Test set evaluation
   - Final pipeline construction

## Key Takeaways

- Demonstrates an end-to-end ML workflow: data cleaning, model building, cross-validation, and deployment-ready pipelines
- Highlights the importance of **aligning evaluation metrics with real business costs**, rather than relying solely on accuracy
- Explores **decision threshold tuning** as a practical lever to balance false alarms against missed failures
- Compares **oversampling vs. undersampling** strategies for handling imbalanced failure data

## Future Improvements

- Experiment with tree-based models (XGBoost, LightGBM) which often perform strongly on tabular sensor data and offer feature importance insights
- Perform feature engineering on raw sensor signals (rolling averages, trend features) to capture degradation patterns over time
- Use threshold optimization techniques (e.g., precision-recall curve analysis) to systematically find the cost-optimal threshold rather than testing fixed values
- Deploy the final pipeline as a real-time scoring API for live sensor data
