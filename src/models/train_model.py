import yaml
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import dagshub
from tqdm import tqdm
from functools import partial

dagshub.init(repo_owner='sarthakg004', repo_name='fraud_detection', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/sarthakg004/fraud_detection.mlflow")

experiment_name = yaml.safe_load(open('./params.yaml'))['paths']['experiment_name']
mlflow.set_experiment(experiment_name=experiment_name)

# Start an MLflow run
with mlflow.start_run(run_name='model_building'):   
    
    '''Loading parameters'''
    paths = yaml.safe_load(open('./params.yaml'))['paths']
    params = yaml.safe_load(open('./params.yaml'))['model_building']
    feature_params = yaml.safe_load(open('./params.yaml'))['feature_engineering']
    preprocess_params = yaml.safe_load(open('./params.yaml'))['preprocessing']
    

    TRAIN_DATA_PATH = paths['train_data_processed_path']
    VALIDATION_DATA_PATH = paths['test_data_processed_path']
    MODEL_PATH = paths['model_path']
    MODEL = params['model']
    TEST_SIZE = params['test_size']
    RANDOM_STATE = params['random_state']
    FEATURE_SELECTION_TECHNIQUE = yaml.safe_load(open('./params.yaml'))['feature_engineering']['selection_technique']
    HYPERPARAMETER_TUNING = params['tuning']
    THRESHOLD = params['threshold']
    XGB_EVAL_METRIC = params['XGB_eval_metric']
    DEVICE = yaml.safe_load(open('./params.yaml'))['feature_engineering']['forward_selection_device']

    

    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_param("feature_selection_technique", FEATURE_SELECTION_TECHNIQUE)
    mlflow.log_param("model_type", MODEL)
    mlflow.log_param("hyperparameter_tuning", HYPERPARAMETER_TUNING)


    
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    validation_df = pd.read_csv(VALIDATION_DATA_PATH)
    
    

    mlflow.log_input(mlflow.data.from_pandas(train_df), "preprocessed_data")
    mlflow.log_input(mlflow.data.from_pandas(validation_df), "validation_data")
    mlflow.log_artifact(TRAIN_DATA_PATH, "preprocessed_data")
    mlflow.log_artifact(VALIDATION_DATA_PATH, "validation_data")
    
    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(train_df.drop(columns=['Target']), train_df['Target'],
                                                        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=train_df['Target'])
    
    

    if MODEL == 'XGB':
        if not HYPERPARAMETER_TUNING:
            # Initialize the XGBoost classifier
            model = XGBClassifier(eval_metric=XGB_EVAL_METRIC, random_state=42)
            # Train the model
            model.fit(X_train, y_train)

            threshold = THRESHOLD
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba >= threshold).astype(int)

            # Calculate various metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')
            roc_auc = roc_auc_score(y_test, y_pred_proba)

            # Print metrics
            print(f'Accuracy on Validation Set: {accuracy:.4f}')
            print(f'Precision on Validation Set: {precision:.4f}')
            print(f'Recall on Validation Set: {recall:.4f}')
            print(f'F1-Score on Validation Set: {f1:.4f}')
            print(f'ROC AUC Score on Validation Set: {roc_auc:.4f}')

            # Log parameters and metrics with MLflow
            mlflow.log_param("threshold", threshold)
            mlflow.log_params(model.get_params())
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc_score", roc_auc)

            # Compute and plot the confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix')
            cm_path = f'./reports/figures/confusion_matrix.png'
            plt.savefig(cm_path)
            mlflow.log_artifact(cm_path)

        else:
            # Define the parameter grid
            param_grid = {
                'n_estimators': [100, 200, 300, 500 , 600],
                'max_depth': [3, 5, 7, 9 , 11],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'gamma': [0, 0.1, 0.2, 0.3],
                'min_child_weight': [1, 3, 5]
            }

            # Initialize XGBoost classifier with GPU support
            xgb_model = XGBClassifier(
                tree_method='hist',   # Use GPU acceleration
                device= DEVICE,
                objective='binary:logistic',
                eval_metric=XGB_EVAL_METRIC,
                random_state=42
            )

            # Define StratifiedKFold cross-validator
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            '''Random Search CV'''
            # Define the RandomizedSearchCV with f1 score as the scoring metric
            # random_search = RandomizedSearchCV(
            #     estimator=xgb_model,
            #     param_distributions=param_grid,
            #     scoring='f1_macro',
            #     n_iter=2000,  # Number of parameter settings that are sampled
            #     cv=cv,
            #     verbose=1,
            #     random_state=42,
            #     n_jobs=-1  # Use all available cores
            # )

            # # Fit the model to the training data
            # random_search.fit(X_train, y_train)

            # # After running the GridSearchCV and obtaining the best model
            # model = random_search.best_estimator_

            
            '''Grid Search CV'''
            
            # Define the GridSearchCV with f1 score as the scoring metric
            grid_search = GridSearchCV(
                estimator=xgb_model,
                param_grid=param_grid,
                scoring='f1_macro',
                cv=cv,
                verbose=0,
                n_jobs=-1  # Use all available cores
            )
            
            # Fit the model to the training data
            grid_search.fit(X_train, y_train)
            
            # After running the GridSearchCV and obtaining the best model
            model = grid_search.best_estimator_
            
            # Adjusting the decision threshold
            threshold = THRESHOLD
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba >= threshold).astype(int)

            # Log parameters and metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')
            roc_auc = roc_auc_score(y_test, y_pred_proba)

            # Print metrics
            print(f'Accuracy on Validation Set: {accuracy:.4f}')
            print(f'Precision on Validation Set: {precision:.4f}')
            print(f'Recall on Validation Set: {recall:.4f}')
            print(f'F1-Score on Validation Set: {f1:.4f}')
            print(f'ROC AUC Score on Validation Set: {roc_auc:.4f}')

            # Log parameters and metrics with MLflow
            mlflow.log_param("threshold", threshold)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc_score", roc_auc)
            
            # for i in range(len(random_search.cv_results_['params'])):
            #     with mlflow.start_run(nested=True):
            #         # Log Hyperparameters
            #         mlflow.log_params(random_search.cv_results_['params'][i])
            #         mlflow.log_metric('mean_f1_test_score', random_search.cv_results_['mean_test_score'][i])
                    
            
            mlflow.log_params(model.get_params())
            
            # Compute and plot the confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix')
            cm_path = f'./reports/figures/confusion_matrix.png'
            plt.savefig(cm_path)
            mlflow.log_artifact(cm_path)
        
        # Save the trained model
        joblib.dump(model, MODEL_PATH)
        mlflow.log_artifact(MODEL_PATH)


    elif MODEL == 'ANOMALY':
        pass



    # Initialize lists to store false positives and false negatives
    false_positives = []
    false_negatives = []
    thresholds = np.arange(0, 1.01, 0.01)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    # Loop over thresholds from 0.01 to 1 with a step of 0.01
    for threshold in thresholds:
        y_pred_threshold = (y_pred_proba >= threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred_threshold)
        
        # Extract false positives and false negatives from the confusion matrix
        FP = cm[0, 1]
        FN = cm[1, 0]
        
        false_positives.append(FP)
        false_negatives.append(FN)

    # Calculate the difference between false positives and false negatives
    diff_fp_fn = np.abs(np.array(false_positives) - np.array(false_negatives))

    # Find the index of the minimum difference
    intersection_index = np.argmin(diff_fp_fn)
    intersection_threshold = thresholds[intersection_index]
    intersection_fp = false_positives[intersection_index]
    intersection_fn = false_negatives[intersection_index]

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, false_positives, label='False Positives', color='red')
    plt.plot(thresholds, false_negatives, label='False Negatives', color='blue')
    # Plot the intersection point
    plt.scatter(intersection_threshold, intersection_fp, color='green', s=100, label=f'Intersection at {intersection_threshold:.2f}')

    plt.xlabel('Threshold')
    plt.ylabel('Count')
    plt.title('False Positives and False Negatives vs Threshold')
    plt.legend(loc='upper right')
    plt.grid(True)
    threshold_path = f'./reports/figures/Falsepositive_FalseNegative.png'
    plt.savefig(threshold_path)
    mlflow.log_artifact(threshold_path)
    
    
    # Save and log the model
    joblib.dump(model, MODEL_PATH)
    mlflow.log_artifact(MODEL_PATH)
    signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
    mlflow.xgboost.log_model(model, "model", signature=signature)
    mlflow.set_tag("model_type", MODEL)

    print("Model saved and logged successfully.")
