import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import joblib



'''Loading parameters'''
paths  =yaml.safe_load(open('./params.yaml'))['paths']
params  =yaml.safe_load(open('./params.yaml'))['model_building']

TRAIN_DATA_PATH = paths['train_data_processed_path']
VALIDATION_DATA_PATH = paths['test_data_processed_path']
MODEL_PATH = paths['model_path']

MODEL = params['model']
TEST_SIZE = params['test_size']
RANDOM_STATE = params['random_state']
FEATURE_SELECTION_TECHNIQUE  =yaml.safe_load(open('./params.yaml'))['feature_engineering']['selection_technique']
HYPERPARAMETER_TUNING = params['tuning']
THRESHOLD = params['threshold']

train_df = pd.read_csv(TRAIN_DATA_PATH)
validation_df = pd.read_csv(VALIDATION_DATA_PATH)


# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(train_df.drop(columns=['Target']),train_df['Target'] ,
                                                    test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=train_df['Target'])



if MODEL == 'XGB':
    if HYPERPARAMETER_TUNING ==False:
        # Initialize the XGBoost classifier
        model = XGBClassifier(eval_metric='logloss', random_state=42)
        # Train the model
        model.fit(X_train, y_train)
        
        threshold = 0.3
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Calculate F1-score
        f1 = f1_score(y_test, y_pred, average='macro')
        print(f'F1-Score on Validation Set: {f1:.4f}')
        
            # Compute the confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        # Plot the confusion matrix using seaborn's heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.savefig(f'./reports/figures/{MODEL}_{FEATURE_SELECTION_TECHNIQUE}')
    
    if HYPERPARAMETER_TUNING == True :

        # Define the parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 0.1, 0.2, 0.3],
            'min_child_weight': [1, 3, 5]
        }

        # Initialize XGBoost classifier with GPU support
        xgb_model = XGBClassifier(
            tree_method='gpu_hist',   # Use GPU acceleration
            predictor='gpu_predictor',
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42
        )

        # Define StratifiedKFold cross-validator
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Define the RandomizedSearchCV with f1 score as the scoring metric
        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_grid,
            scoring='f1_macro',
            n_iter=50,  # Number of parameter settings that are sampled
            cv=cv,
            verbose=1,
            random_state=42,
            n_jobs=-1  # Use all available cores
        )

        # Fit the model to the training data
        random_search.fit(X_train, y_train)
        
        # After running the RandomizedSearchCV and obtaining the best model
        model = random_search.best_estimator_

        # Adjusting the decision threshold
        ## threshold for transaction to be non fradulent i.e. if threshold = 1 all transactions are non fradulent
        threshold = THRESHOLD
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred_threshold = (y_pred_proba >= threshold).astype(int)

        # Calculate F1-score
        f1 = f1_score(y_test, y_pred_threshold, average='macro')
        print(f'F1-Score on Validation Set: {f1:.4f}')

        cm = confusion_matrix(y_test, y_pred)
        # Plot the confusion matrix using seaborn's heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.savefig(f'./reports/figures/{MODEL}_{FEATURE_SELECTION_TECHNIQUE}')
    
    
elif MODEL == 'ANOMOLY':
    pass



# Save the model
joblib.dump(model, MODEL_PATH)


