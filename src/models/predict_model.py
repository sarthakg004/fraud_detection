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
params  =yaml.safe_load(open('./params.yaml'))['predict_model']

MODEL_TYPE = yaml.safe_load(open('./params.yaml'))['model_building']['model']
THRESHOLD = yaml.safe_load(open('./params.yaml'))['model_building']['threshold']
MODEL_PATH = paths['model_path']
INPUT = params['input']


model = joblib.load(MODEL_PATH)




# Convert the user input to a DataFrame
if INPUT == 'manual':
    pass
    # Convert the user input to a DataFrame
    # df = pd.DataFrame(user_input_data)
    
elif INPUT == 'file':
    # path = (input('Enter the path to the file: '))
    # df = pd.read_csv(path)
    df  =pd.read_csv('./data/processed/validation_data_processed.csv')
    
if MODEL_TYPE == 'XGB':
    threshold = THRESHOLD
    y_pred_proba = model.predict_proba(df)[: , 1]
    y_pred_threshold = (y_pred_proba >= threshold).astype(int)
    pd.Series(y_pred_threshold).to_csv('./predictions.csv', index=False)
    

