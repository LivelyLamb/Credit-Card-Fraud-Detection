* CREDIT CARD Fraud Detection System

An end to end machine learning system for detecting frauds in credit card transactions using supervised and unsupervised learning.

The project explores highly imbalanced dataset, from kaggle.
It does so through model comparisions, hyperparameter tuning, feaure engineering, threshold optimization as well as anomaly detection.
The final model is exposed through FASTAPI REST which accepts raw transaction data and returns its classification i.e Fraud or Legit and probability by which it was classified.

* PROJECT HIGHLIGHTS :)
1. Explored a very highly imbalanced credit card fraud dataset.
2. Built a logistic regression baseline.
3. Compared Random Forest and XG Boost models
4. Performed XG Boost hyperparameter tuning using validation set.
5. Preservd the the training preprocessing pipeline for inference
6. Seralized the trained model and scaler.
7. Built and tested API a FASTAPI prediction APII
8. Tested API using swagger using

-------------------------------------------------------------------------------------***-----------------------------------------------------------------------

* FINAL MODEL

The final XG Boost Model achieved:

ROC-AUC: 0.9713
PR-AUC: 0.8747
Precision (Fraud): 0.92
Recall (Fraud): 0.81
F1: 0.86

The classification threshold was selected on the validation set and then applied to the untouched test set.

-------------------------------------------------------------------------------------***-----------------------------------------------------------------------

* DATASET 

Credit Card Fraud Detection dataset from Kaggle.
The dataset contains 284,807 transactions with 31 columns.
Time: Seconds elapsed between each transaction and the first transaction in the dataset.
Amount: transaction Amount
V1 to V28: Anonymized features obtained through PCA transformation.
Class: Target variable where 0 represents a legitimate transaction and 1 represents fraud.

Contains only 492 fraudulent transactions.
This makes the problem highly imbalanced and means that accuracy alone is not a useful metric for judging the model.

-------------------------------------------------------------------------------------***-----------------------------------------------------------------------

* PROBLEM STATEMENT

The main goal of the project is to identify fraudulent transactions while keeping false alarms under control.
A model that predicts every transaction as legitimate can achieve very high accuracy because fraudulent transactions are extremely rare.
However, such a model would completely fail at the actual task of detecting fraud.
Because of this, the project focuses more on Precision, Recall,
F1-Score, ROC-AUC and PR-AUC rather than relying only on accuracy.
There is also a trade-off between catching more fraudulent transactions and generating more false positives.
This is why the classification threshold was treated as a part of the fraud detection system rather than simply using the default 0.5 threshold.

-------------------------------------------------------------------------------------***-----------------------------------------------------------------------

* FEATURE ENGINEERING AND preprocessing

The dataset contained no null or empty values. So no amputation of those vales was required.
The data was split into training and testing set using stratyfing due to the actual fraud being just 0.017% of the data.
By stratifying the ratio of fraud was preserved.

-------------------------------------------------------------------------------------***-----------------------------------------------------------------------

* FEATURE SCALING

The time and amount features were scaled using Standardscaler().
The features V1 - V28 were already Anonymized and transformed features so were not included into scaling.
The scaler was fitten only on the training data and then used to transform validation set and test data.
Tis done so, to avoid leaking of information from training data process to testing data.

-------------------------------------------------------------------------------------***-----------------------------------------------------------------------

* LOG transformation

The amount feature was highly skewed with number of transaction having very large values.
To reduce this skew, a new feature was created using:
LogAmount = log1p(Amount)
The raw Amount, Time and LogAmount were then scaled.
The final preprocessing pipeline was:
Amount -> LogAmount -> Scaling -> Model 
This same preprocessing pipeline was saved and reused when making predictions through the FastAPI API.

-------------------------------------------------------------------------------------***-----------------------------------------------------------------------

* MODEL DEVELOPMENT & COMPARISON

I started with Logistic Regression as a baseline model before moving to tree-based models.

* 1. Logistic Regression

Logistic Regression was used as the initial baseline because it provides a simple and interpretable reference point.

After feature engineering, the model achieved:

- ROC-AUC: 0.8264
- PR-AUC: 0.5434
- Precision (Fraud): 0.83
- Recall (Fraud): 0.65
- F1-Score: 0.73

The results showed that a simple linear model could detect some fraudulent transactions, but there was room for improvement.

*_________________________________________________________________________*

* 2. Random Forest

Random Forest was then tested to capture more complex, non-linear relationships in the data.

It achieved:

- ROC-AUC: 0.9528
- PR-AUC: 0.8762
- Precision (Fraud): 0.95
- Recall (Fraud): 0.82
- F1-Score: 0.88

This was a large improvement over the Logistic Regression baseline.

*_________________________________________________________________________*

* 3. XGBoost

XGBoost was then evaluated as another tree-based approach.

The initial XGBoost model achieved:

- ROC-AUC: 0.9612
- PR-AUC: 0.8253
- Precision (Fraud): 0.83
- Recall (Fraud): 0.65
- F1-Score: 0.73

Although the initial XGBoost model had a higher ROC-AUC than Random Forest, its PR-AUC and default-threshold classification results were worse.
This showed why relying on a single metric can give an incomplete picture.

-------------------------------------------------------------------------------------***-----------------------------------------------------------------------

* HYPERPARAMETER TUNING

XGBoost hyperparameters were tuned using a separate validation set rather than repeatedly evaluating different configurations on the test set.

The main parameters explored were:

max_depth
learning_rate
n_estimators

The final configuration selected using validation performance was:

max_depth = 10
learning_rate = 0.05
n_estimators = 400

The final model was then retrained using the full training set and evaluated once on the untouched test set.

This produced:

ROC-AUC: 0.9713
PR-AUC: 0.8747
Precision (Fraud): 0.92
Recall (Fraud): 0.81
F1-Score (Fraud): 0.86

-------------------------------------------------------------------------------------***-----------------------------------------------------------------------

## HANDLING CLASS IMBALANCE

Fraud represented only a very small fraction of the transactions, so class imbalance was a major consideration throughout the project.
Experimented with XGBoost's \scale_pos_weight\ parameter using the negative-to-positive class ratio:

scale_pos_weight is around 577.65

The weighted model was compared against the unweighted model on the validation set.
In this experiment, class weighting did not improve the overall results.
The weighted model caught slightly more fraud cases but also produced more false positives and had lower PR-AUC and F1-Score.
Instead of forcing class weighting into the final model, I kept the unweighted model and investigated threshold optimization separately.

-------------------------------------------------------------------------------------***-----------------------------------------------------------------------

* THRESHOLD OPTIMIZATION

The default classification threshold of a binary classifier is usually `0.5`, but this is not necessarily the best choice for a fraud detection system.
The final XGBoost model produces a fraud probability. A threshold is then used to convert that probability into a final classification.
I tested different thresholds on the validation set and selected **0.31** based on the F1-Score.
At this threshold, the validation set achieved:

Precision: 0.925
Recall: 0.785
F1-Score: 0.849

The selected threshold was then applied once to the untouched test set.

The resulting test performance was:

Precision: 0.90
Recall: 0.83
F1-Score: 0.86
False Positives: 9
False Negatives: 17

This demonstrates the trade-off involved in fraud detection. Lowering the threshold can catch more fraudulent transactions, but it can also increase the number of legitimate transactions incorrectly flagged as fraud.

-------------------------------------------------------------------------------------***-----------------------------------------------------------------------

* UNSUPERVISED ANOMALY DETECTION

Also experimented with Isolation Forest to investigate whether an unsupervised anomaly detection approach could identify fraudulent transactions without using the fraud labels during training.
Different contamination values were tested rather than assuming that the fraud percentage was automatically the correct anomaly percentage.
The results were significantly worse than the supervised XGBoost model.
However, the experiment was useful because it showed that an anomaly detector does not automatically identify fraud simply because fraudulent transactions are rare.
Unusual legitimate transactions can also be classified as anomalies.
For this project, Isolation Forest was therefore treated as an additional experiment rather than the final fraud detection model.

-------------------------------------------------------------------------------------***-----------------------------------------------------------------------aaa

* API and DEPLOYMENT

The trained model was integrated into a FastAPI REST API so that it can be used outside of the training notebook.
The API accepts raw transaction data, performs the same preprocessing used during training, and returns the fraud probability and final classification.

Raw transaction  -> FastAPI request validation -> Create DataFrame -> Create LogAmount -> Apply saved StandardScaler -> Arrange features in training order-> XGBoost model -> Fraud probability -> Threshold = 0.31 -> Fraud / Legitimate

ENDPOINT: 

The endpoint accepts a JSON transaction containing:

Time, V1 to V28, Amount

The API was tested using FastAPI's Swagger UI.
The API was also exposed through a temporary development tunnel so that it could be accessed remotely for testing.

* MODEL INFERENCE

The trained model and preprocessing objects are included separately from the notebook so that predictions can be made without retraining the model.

-------------------------------------------------------------------------------------***-----------------------------------------------------------------------

* LIMITATIONS AND FUTURE IMPROVEMENTS 

Although the model works well on the given dataset, there are several LIMITATIONS.

The data contains Anonymized V1 to V28 and their real-world meaning cannot be interpreted.
The dataset is relatively old and may not contain current fraud patterns.
The classification threshold of 0.31 was optimised for this dataset and may need caliberations for a real Fraud Detection.
The API currently processes individual requests or transactions than a complete batchs or streaming transactions.
The model does not include the additional real-world information such as customer history, transaction history and much more,
which could help the classification even more precise.

Possible future improvements include:

Monitoring model performance over time and detecting data drift.
Retraining the model as new fraud patterns appear.
Adding batch prediction support.
Adding authentication and rate limiting to the API.
Deploying the API to a cloud platform.
Building a frontend dashboard for monitoring predictions and fraud statistics.
Combining supervised fraud prediction with anomaly detection signals.