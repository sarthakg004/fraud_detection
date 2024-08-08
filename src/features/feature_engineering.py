import pandas as pd
import yaml
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SequentialFeatureSelector

'''Loading parameters'''
paths  =yaml.safe_load(open('./params.yaml'))['paths']
params  =yaml.safe_load(open('./params.yaml'))['feature_engineering']

TRAIN_DATA_PATH = paths['train_data_cleaned_path']
VALIDATION_DATA_PATH = paths['test_data_cleaned_path']
TRAIN_DATA_PROCESSED_PATH = paths['train_data_processed_path']
VALIDATION_DATA_PROCESSED_PATH = paths['test_data_processed_path']
VARIANCE_THRESHOLD = params['variance_threshold']
SELECTION_TECHNIQUE = params['selection_technique']
RANDOM_STATE = params['random_state']
KFOLDS = params['k_folds']
REFCV_RF_STEP = params['rfecv_rf_step']
REFCV_RF_SCORING = params['rfecv_rf_scoring']
REFCV_XGB_EVAL_METRIC = params['rfecv_xgb_eval_metric']
REFCV_XGB_STEP = params['rfecv_xgb_step']
REFCV_XGB_SCORING = params['rfecv_xgb_scoring']
ANOVA_K = params['anova_k']
ANOVA_FIND_K = params['anova_find_k']
SFS_DIRECTION = params['sfs_direction']
FORWARD_SELECTION_CLASSIFIER = params['forward_selection_classifier']
FORWARD_SELECTION_DEVICE = params['forward_selection_device']
FORWARD_SELECTION_N_FEATURES = params['forward_selection_n_features']
FORWARD_SELECTION_SCORING = params['forward_selection_scoring']
FORWARD_SELECTION_CV = params['forward_selection_cv']
FORWARD_SELECTION_XGB_EVAL_METRIC = params['forward_selection_xgb_eval_metric']

'''Loading data'''
train_df = pd.read_csv(TRAIN_DATA_PATH)
validation_df = pd.read_csv(VALIDATION_DATA_PATH)

X = train_df.drop(columns='Target')
y = train_df['Target']


'''Removing constant features'''
# Initialize VarianceThreshold with a threshold of 0 to remove constant features
selector = VarianceThreshold(threshold=VARIANCE_THRESHOLD)

selector.fit_transform(X)

# If you want to keep the selected feature names:
selected_features = X.columns[selector.get_support(indices=True)]
X = X[selected_features]
validation_df = validation_df[selected_features]


'''Performing feature selection based on selection technique'''
if SELECTION_TECHNIQUE == 'ANOVA':
    if ANOVA_FIND_K : 

        # Initialize variables
        k_values = list(range(1, X.shape[1] + 1))  # Testing all possible values of k
        mean_scores = []

        # Perform cross-validation for each k value
        for k in k_values:
            selector = SelectKBest(score_func=f_classif, k=k)
            X_new = selector.fit_transform(X, y)
            # Initialize classifier (e.g., Random Forest)
            clf = RandomForestClassifier(random_state=42)
            # Perform cross-validation
            cv = StratifiedKFold(n_splits=5)
            scores = cross_val_score(clf, X_new, y, cv=cv, scoring='f1_macro')  # or any other metric
            mean_scores.append(np.mean(scores))
        # Find the optimal number of features
        optimal_k = k_values[np.argmax(mean_scores)]
        selector = SelectKBest(score_func=f_classif, k=optimal_k).fit(X, y)
        selected_features = X.columns[selector.get_support()]

    else:
        selector = SelectKBest(score_func=f_classif, k=ANOVA_K).fit(X, y)
        selected_features = X.columns[selector.get_support()]
    
    X = X[selected_features]
    validation_df = validation_df[selected_features]   

elif SELECTION_TECHNIQUE == 'RFECV_RF':
    rf = RandomForestClassifier(random_state=RANDOM_STATE)

    # Initialize RFECV with cross-validation
    rfecv_rf = RFECV(estimator=rf, step=REFCV_RF_STEP, cv=StratifiedKFold(KFOLDS), scoring=REFCV_RF_SCORING)

    # Fit RFECV
    rfecv_rf.fit(X, y)

    selected_features_rf = X.columns[rfecv_rf.support_]
    X = X[selected_features_rf]
    validation_df = validation_df[selected_features_rf]
    
        # Optimal number of features
    optimal_features = rfecv_rf.n_features_
    print(f'Optimal number of features: {optimal_features}')

elif SELECTION_TECHNIQUE == 'RFECV_XGB':
    # Initialize XGBClassifier
    xgb = XGBClassifier(random_state=RANDOM_STATE, eval_metric=REFCV_XGB_EVAL_METRIC)

    # Initialize RFECV with cross-validation
    rfecv_xgb = RFECV(estimator=xgb, step=REFCV_XGB_STEP, cv=StratifiedKFold(KFOLDS), scoring=REFCV_RF_SCORING)
    # Fit RFECV
    rfecv_xgb.fit(X, y)
    
    selected_features_xgb = X.columns[rfecv_xgb.support_]
    X = X[selected_features_xgb]
    validation_df = validation_df[selected_features_xgb]
    
    # Optimal number of features
    optimal_features = rfecv_xgb.n_features_
    print(f'Optimal number of features: {optimal_features}')


elif SELECTION_TECHNIQUE == 'FORWARD_SELECTION':
    # Initialize classifier
    clf = {"RF": RandomForestClassifier(random_state=RANDOM_STATE,n_jobs=-1),    
           "XGB": XGBClassifier(random_state=RANDOM_STATE,device = FORWARD_SELECTION_DEVICE,eval_metric=FORWARD_SELECTION_XGB_EVAL_METRIC)}[FORWARD_SELECTION_CLASSIFIER]

    # Initialize SequentialFeatureSelector
    sfs = SequentialFeatureSelector(
        clf,
        n_features_to_select=FORWARD_SELECTION_N_FEATURES,  # or specify the number of features, e.g., 10
        direction=SFS_DIRECTION,
        scoring=FORWARD_SELECTION_SCORING,  # or 'f1' for f1-score, etc.
        cv=FORWARD_SELECTION_CV,  # Number of folds in cross-validation
        n_jobs=-1  # Use all available cores
    )

    # Fit the SFS to the data
    sfs.fit(X, y)

    # Get the selected feature indices
    selected_features_sfs = X.columns[sfs.get_support()]
    X = X[selected_features_sfs]
    validation_df = validation_df[selected_features_sfs]

    print(f'Optimal number of features: {sfs.n_features_in_}')

    

'''Saving processed data'''
train_df = pd.concat([X, y], axis=1)
train_df.to_csv(TRAIN_DATA_PROCESSED_PATH, index=False)
validation_df.to_csv(VALIDATION_DATA_PROCESSED_PATH, index=False)