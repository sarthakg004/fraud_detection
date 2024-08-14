import pandas as pd
import yaml
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder
import os
import mlflow
import mlflow.pyfunc

# Start MLflow run
with mlflow.start_run(run_name="data_preprocessing"):

    paths  = yaml.safe_load(open('./params.yaml'))['paths']
    params  = yaml.safe_load(open('./params.yaml'))['preprocessing']

    # Log parameters
    mlflow.log_params(params)
    
    TRAIN_DATA_PATH = paths['raw_train_data_path']
    VALIDATION_DATA_PATH = paths['raw_test_data_path']
    CLEAN_TRAIN_DATA_PATH = paths['train_data_cleaned_path']
    CLEAN_VALIDATION_DATA_PATH = paths['test_data_cleaned_path']
    NULL_PERCENT = params['null_percentage']
    NULL_ROW_THRESHOLD = params['null_row_threshold']
    ENCODING_TYPE = params['encoding_type']
    IMPUTATON_TYPE = params['imputation_strategy']
    KNN_N_NEIGHBORS = params['knn_n_neighbors']

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    validation_df = pd.read_csv(VALIDATION_DATA_PATH)

    # Dropping primary key column
    train_df.drop(columns=['Primary key'], inplace=True)
    validation_df.drop(columns=['Primary key'], inplace=True)

    # Dropping columns with more than NULL_PERCENT missing values
    col_to_drop =  train_df.iloc[:,train_df.isnull().sum().values > NULL_PERCENT*len(train_df)].columns
    train_df.drop(columns = col_to_drop, inplace = True)
    validation_df.drop(columns = col_to_drop, inplace = True)

    # Dropping rows with more than 15 missing values
    train_df['null_count'] = train_df.isnull().sum(axis=1)
    train_df.drop(index = train_df.loc[train_df['null_count'] > NULL_ROW_THRESHOLD].index,inplace=True)
    train_df.drop(columns = 'null_count', inplace = True)

    # Converting date columns to datetime
    train_df['account_opening_date'] = pd.to_datetime(train_df['account_opening_date'], format='%d-%m-%Y')  
    validation_df['account_opening_date'] = pd.to_datetime(validation_df['account_opening_date'], format='%d-%m-%Y')

    # Create new columns for account opening month and year
    train_df['account_opening_month'] = train_df['account_opening_date'].dt.month_name()
    train_df['account_opening_year'] = train_df['account_opening_date'].dt.year
    train_df.drop(columns='account_opening_date', inplace=True)

    validation_df['account_opening_month'] = validation_df['account_opening_date'].dt.month_name()
    validation_df['account_opening_year'] = validation_df['account_opening_date'].dt.year
    validation_df.drop(columns='account_opening_date', inplace=True)

    train_df['account_opening_year'] = train_df['account_opening_year'].astype(str)
    validation_df['account_opening_year'] = validation_df['account_opening_year'].astype(str)

    '''Converting the 'income' column to a categorical column with the following categories:'''
    def map_to_category(range_str):
        if isinstance(range_str, float):
            return 'Very Low'
        if '0 to 1L' in range_str:
            return 'Very Low'
        if '100001 to 5L' in range_str:
            return 'Low'
        if '5L to 10L' in range_str:
            return 'Medium'
        else:
            return 'High'

    train_df['income'] = train_df['income'].astype(str)
    train_df['income_category'] = train_df['income'].map(map_to_category)
    train_df.drop(columns = 'income', inplace = True)

    validation_df['income'] = validation_df['income'].astype(str)
    validation_df['income_category'] = validation_df['income'].map(map_to_category) 
    validation_df.drop(columns = 'income', inplace = True)  

    '''Converting the 'country_code' column to a categorical column with the following categories:'''
    def map_to_category(range_str):
        if isinstance(range_str, str) and 'IN' in range_str:
            return 'INDIA'
        else:
            return 'OTHER'

    train_df['country_code'] = train_df['country_code'].astype(str)
    train_df['country_code'] = train_df['country_code'].map(map_to_category)

    validation_df['country_code'] = validation_df['country_code'].astype(str)
    validation_df['country_code'] = validation_df['country_code'].map(map_to_category)

    # Dropping rows with 'ZZ' in 'demog_2' column
    train_df.drop(index = train_df[train_df['demog_2'] == 'ZZ'].index, inplace = True)
    validation_df.drop(index = validation_df[validation_df['demog_2'] == 'ZZ'].index, inplace = True)
    train_df['demog_2'] = train_df['demog_2'].astype(float)
    validation_df['demog_2'] = validation_df['demog_2'].astype(float)

    # Dropping rows with 'ZZ' in 'demog_3' column
    train_df.loc[train_df['demog_4'] == 'N', 'demog_4'] = 0
    train_df['demog_4'] = train_df['demog_4'].astype(float)

    validation_df.loc[validation_df['demog_4'] == 'N', 'demog_4'] = 0
    validation_df['demog_4'] = validation_df['demog_4'].astype(float)

    train_df.drop(columns = ['demog_10'], inplace = True)
    validation_df.drop(columns = ['demog_10'], inplace = True)

    train_df.drop(columns = ['demog_22'], inplace = True)
    validation_df.drop(columns = ['demog_22'], inplace = True)

    '''Converting boolean columns to boolean type'''
    bool_col = ['demog_9'	,'demog_12'	,'demog_13'	,'demog_14'	,'demog_15'	,'demog_16'	,'demog_17'	,'demog_18'	,'demog_19','demog_20','demog_21']
    for i in bool_col:
        train_df[i] = train_df[i].astype(bool)
        validation_df[i] = validation_df[i].astype(bool)

    for col in bool_col:
        train_df[col] = train_df[col].map({True: 1, False: 0})
        validation_df[col] = validation_df[col].map({True: 1, False: 0})
        
    '''Converting the 'email_domain' column to a categorical column with the following categories:'''
    def map_to_category(range_str):
        if isinstance(range_str, str) and 'gmail' in range_str:
            return 'GMAIL'
        else:
            return 'OTHER'

    train_df['email_domain'] = train_df['email_domain'].astype(str)
    train_df['email_domain'] = train_df['email_domain'].map(map_to_category)

    validation_df['email_domain'] = validation_df['email_domain'].astype(str)
    validation_df['email_domain'] = validation_df['email_domain'].map(map_to_category)

    '''encoding categorical columns'''
    categorical_columns = train_df.select_dtypes(include=['object']).columns
    if ENCODING_TYPE == "frequency":
        ### frequency encoding categorical columns
        for col in categorical_columns:
            freq_encoding = train_df[col].value_counts() / len(train_df)
            train_df[col] = train_df[col].map(freq_encoding)
            
            freq_encoding = validation_df[col].value_counts() / len(validation_df)
            validation_df[col] = validation_df[col].map(freq_encoding)
    if ENCODING_TYPE == "ordinal":      
        ordinal_encoder = OrdinalEncoder()
        train_df[categorical_columns] = ordinal_encoder.fit_transform(train_df[categorical_columns])
        validation_df[categorical_columns] = ordinal_encoder.transform(validation_df[categorical_columns])

    '''Imputing missing values'''
    if IMPUTATON_TYPE == "KNN":
        '''Imputing missing values using KNNImputer'''
        train_exclude = train_df[['account_opening_date', 'Target']]
        validation_exclude = validation_df[['account_opening_date']]

        train_features = train_df.drop(columns=['account_opening_date', 'Target'])
        validation_features = validation_df.drop(columns=['account_opening_date'])

        imputer = KNNImputer(n_neighbors=KNN_N_NEIGHBORS)

        train_imputed = pd.DataFrame(imputer.fit_transform(train_features), columns=train_features.columns)
        validation_imputed = pd.DataFrame(imputer.transform(validation_features), columns=validation_features.columns)

        train_imputed = pd.concat([train_imputed, train_exclude.reset_index(drop=True)], axis=1)
        validation_imputed = pd.concat([validation_imputed, validation_exclude.reset_index(drop=True)], axis=1)
    else:
        '''Imputing missing values using SimpleImputer'''
        from sklearn.impute import SimpleImputer

        train_exclude = train_df[['Target']]
        train_features = train_df.drop(columns=['Target'])

        imputer = SimpleImputer(strategy=IMPUTATON_TYPE)

        train_imputed = pd.DataFrame(imputer.fit_transform(train_features), columns=train_features.columns)
        validation_imputed = pd.DataFrame(imputer.transform(validation_df), columns=validation_df.columns)

        train_imputed = pd.concat([train_imputed, train_exclude.reset_index(drop=True)], axis=1)
    
    # Save preprocessed data
    train_imputed.to_csv(CLEAN_TRAIN_DATA_PATH, index=False)
    validation_imputed.to_csv(CLEAN_VALIDATION_DATA_PATH, index=False)
    
    # Log the cleaned datasets as artifacts
    mlflow.log_artifact(CLEAN_TRAIN_DATA_PATH, artifact_path="cleaned_data")
    mlflow.log_artifact(CLEAN_VALIDATION_DATA_PATH, artifact_path="cleaned_data")

    mlflow.end_run()
