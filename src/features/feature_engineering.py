import pandas as pd
import yaml
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, RFECV, SequentialFeatureSelector
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import mlflow
import matplotlib.pyplot as plt
import mlflow.sklearn
import dagshub


dagshub.init(repo_owner='sarthakg004', repo_name='fraud_detection', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/sarthakg004/fraud_detection.mlflow")

experiment_name = yaml.safe_load(open('./params.yaml'))['paths']['experiment_name']
mlflow.set_experiment(experiment_name=experiment_name)
# Start MLflow run
with mlflow.start_run(run_name="feature_engineering"):

    '''Loading parameters'''
    paths = yaml.safe_load(open('./params.yaml'))['paths']
    params = yaml.safe_load(open('./params.yaml'))['feature_engineering']



    TRAIN_DATA_PATH = paths['train_data_cleaned_path']
    VALIDATION_DATA_PATH = paths['test_data_cleaned_path']
    TRAIN_DATA_PROCESSED_PATH = paths['train_data_processed_path']
    VALIDATION_DATA_PROCESSED_PATH = paths['test_data_processed_path']
    VARIANCE_THRESHOLD = params['variance_threshold']
    SELECTION_TECHNIQUE = params['selection_technique']
    RANDOM_STATE = params['random_state']
    KFOLDS = params['k_folds']
    RFECV_RF_STEP = params['rfecv_rf_step']
    RFECV_RF_SCORING = params['rfecv_rf_scoring']
    RFECV_XGB_EVAL_METRIC = params['rfecv_xgb_eval_metric']
    RFECV_XGB_STEP = params['rfecv_xgb_step']
    RFECV_XGB_SCORING = params['rfecv_xgb_scoring']
    ANOVA_K = params['anova_k']
    ANOVA_FIND_K = params['anova_find_k']
    SFS_DIRECTION = params['sfs_direction']
    FORWARD_SELECTION_CLASSIFIER = params['forward_selection_classifier']
    FORWARD_SELECTION_DEVICE = params['forward_selection_device']
    FORWARD_SELECTION_N_FEATURES = params['forward_selection_n_features']
    FORWARD_SELECTION_SCORING = params['forward_selection_scoring']
    FORWARD_SELECTION_CV = params['forward_selection_cv']
    FORWARD_SELECTION_XGB_EVAL_METRIC = params['forward_selection_xgb_eval_metric']

    
    mlflow.log_param("variance_threshold", VARIANCE_THRESHOLD)
    
    
    '''Loading data'''
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    validation_df = pd.read_csv(VALIDATION_DATA_PATH)

    X = train_df.drop(columns='Target')
    y = train_df['Target']

    '''Removing constant features'''
    selector = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    selector.fit_transform(X)
    selected_features = X.columns[selector.get_support(indices=True)]
    X = X[selected_features]
    validation_df = validation_df[selected_features]
    


    '''Performing feature selection based on selection technique'''
    if SELECTION_TECHNIQUE == 'ANOVA':
        if ANOVA_FIND_K:
            k_values = list(range(1, X.shape[1] + 1))
            mean_scores = []
            for k in k_values:
                selector = SelectKBest(score_func=f_classif, k=k)
                X_new = selector.fit_transform(X, y)
                clf = RandomForestClassifier(random_state=42)
                cv = StratifiedKFold(n_splits=5)
                scores = cross_val_score(clf, X_new, y, cv=cv, scoring='f1_macro')
                mean_scores.append(np.mean(scores))
                
            optimal_k = k_values[np.argmax(mean_scores)]
            
            # Plot the results
            plt.figure(figsize=(10, 6))
            plt.plot(k_values, mean_scores, marker='o')
            plt.xlabel('Number of Features (k)')
            plt.ylabel('Cross-Validated F1 Score (Macro)')
            plt.title('F1_Score_Macro vs. Number of Features')
            path = f'./reports/figures/f1Macro_vs_features.png'
            plt.savefig(path)
            mlflow.log_artifact(path)
            selector = SelectKBest(score_func=f_classif, k=optimal_k).fit(X, y)
            selected_features = X.columns[selector.get_support()]

        else:
            selector = SelectKBest(score_func=f_classif, k=ANOVA_K).fit(X, y)
            selected_features = X.columns[selector.get_support()]

        
        X = X[selected_features]
        validation_df = validation_df[selected_features]    

    elif SELECTION_TECHNIQUE == 'RFECV_RF':
        rf = RandomForestClassifier(random_state=RANDOM_STATE)
        rfecv_rf = RFECV(estimator=rf, step=RFECV_RF_STEP, cv=StratifiedKFold(KFOLDS), scoring=RFECV_RF_SCORING)
        rfecv_rf.fit(X, y)
        selected_features = X.columns[rfecv_rf.support_]
        X = X[selected_features]
        validation_df = validation_df[selected_features]
        optimal_features = rfecv_rf.n_features_
 

    elif SELECTION_TECHNIQUE == 'RFECV_XGB':
        xgb = XGBClassifier(random_state=RANDOM_STATE, eval_metric=RFECV_XGB_EVAL_METRIC)
        rfecv_xgb = RFECV(estimator=xgb, step=RFECV_XGB_STEP, cv=StratifiedKFold(KFOLDS), scoring=RFECV_XGB_SCORING)
        rfecv_xgb.fit(X, y)
        selected_features = X.columns[rfecv_xgb.support_]
        X = X[selected_features]
        validation_df = validation_df[selected_features]
        optimal_features = rfecv_xgb.n_features_


    elif SELECTION_TECHNIQUE == 'FORWARD_SELECTION':
        clf = {"RF": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
               "XGB": XGBClassifier(random_state=RANDOM_STATE, device=FORWARD_SELECTION_DEVICE, eval_metric=FORWARD_SELECTION_XGB_EVAL_METRIC)}[FORWARD_SELECTION_CLASSIFIER]
        sfs = SequentialFeatureSelector(clf, n_features_to_select=FORWARD_SELECTION_N_FEATURES, direction=SFS_DIRECTION,
                                        scoring=FORWARD_SELECTION_SCORING, cv=FORWARD_SELECTION_CV, n_jobs=-1)
        sfs.fit(X, y)
        selected_features = X.columns[sfs.get_support()]
        X = X[selected_features]
        validation_df = validation_df[selected_features]

    
    mlflow.log_param("selected_features", selected_features)
    
    '''Saving processed data'''
    train_df = pd.concat([X, y], axis=1)
    train_df.to_csv(TRAIN_DATA_PROCESSED_PATH, index=False)
    validation_df.to_csv(VALIDATION_DATA_PROCESSED_PATH, index=False)

    mlflow.end_run()
