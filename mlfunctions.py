from sklearn.metrics import make_scorer
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, classification_report, recall_score, roc_auc_score, precision_score
import pandas as pd
pd.set_option('display.max_columns', None)
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import xgboost
import matplotlib.pylab as pl
import shap

shap.initjs()
#########################################
########## DATA PREPROCESSING ###########
#########################################
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


def import_data(file):
    """create a dataframe and optimize its memory usage"""
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)
    df = reduce_mem_usage(df)
    return df

# Function to calculate missing values by column# Funct 
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns



def dataCleaning(data, colsToDrop, targetCol):

    
    print('\nInital Data Shape: ', data.shape)
    train = data.select_dtypes(['number'])
    print('Shape after selecting only numeric labels: ', train.shape)
    display(missing_values_table(train))
    train = train.dropna(how='any', axis=1)
    print('\nShape after dropping columns with nulls ', train.shape)
    X = train.drop([targetCol], axis=1)   
    X = train.drop(colsToDrop, axis=1)
    y = data[targetCol]
    print('\nTypes of target and counts: \n', y.value_counts())

    return X, y



def feature_selection(model, num_of_feats, X, y):

    
    sel_ = SelectFromModel(model, max_features=num_of_feats)
    sel_.fit(X,y)
    
    selected_feat = X.columns[(sel_.get_support())]
    
    print('Total features:', X.shape[1])
    print('Selected features: ', len(selected_feat))
    
    print('Selected features: ', selected_feat) 

    print('Number of R^2 =0 feautures: ',np.sum(sel_.estimator_.coef_ ==0))
    
    return selected_feat


def synthetic_upsampling(X, y):
    sm = SMOTE()
    print("Value counts before: \n", y.value_counts())
    a, b = sm.fit_sample(X, y)
    print("Value counts after: \n", b.value_counts())
    return a, b
# Making pr_auc metric for GridSearchCV
def pr_auc(y, y_pred):
    precision, recall, thresholds = precision_recall_curve(y, y_pred, pos_label=1)
    return auc(recall, precision)
        
my_prauc = make_scorer(pr_auc, greater_is_better=True, needs_proba=False)

#######################################
######### Performance Report ##########
########################################
def perf_data(model, X, y):
    
    y_pred = model.predict(X)

    precision, recall, thresholds = precision_recall_curve(y, y_pred)
    area = auc(recall, precision)
    
    roc_auc = roc_auc_score(y, y_pred)
    ps = precision_score(y.values, y_pred)
    rs = recall_score(y.values, y_pred)
    print("ROC AUC: ", roc_auc)
    print("PR AUC: ",area)
    print("Precision Score: ", ps)
    print("Recall Score: ", rs) 
    print('Confusion Matrix: \n TP, FP \n FN, TN \n', confusion_matrix(y,y_pred))
    print('Classification Report:\n', classification_report(y,y_pred))


def performance_report(model,X_train,y_train,  X_test, y_test):
    print('------------ Performance Report ---------------')
    print('--------------- Training Set ------------------')
    perf_data(model,X_train, y_train)
    print('------------------- Test Set -------------------')
    perf_data(model,X_test, y_test)
    print('------------ End Performance Report ------------')

