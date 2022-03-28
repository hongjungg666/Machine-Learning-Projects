import numpy as np
import pandas as pd


from sklearn.model_selection import KFold

def reg_target_encoding(train, col="device_type", target="click", splits=5):
    """ Computes regularize mean encoding.
    Inputs:
       train: training dataframe
       
    """
    kf = KFold(n_splits=splits, shuffle=False)
    new_col = col + "_" + "mean_enc"
    ### BEGIN SOLUTION
    train[new_col] = np.nan
    for train_index, val_index in kf.split(train[col]):
        X_tr, X_val = train.iloc[train_index], train.iloc[val_index]
        mean_test = X_tr[target].groupby(X_tr[col]).mean().to_dict()
        #calculate the mean group by categorical col
        train.loc[train.index[val_index], new_col] = X_val[col].map(mean_test)
    train[new_col].fillna(train[target].mean(),inplace=True)
    return train
    ### END SOLUTION



def mean_encoding_test(test, train, col="device_type", target="click"):
    """ Computes target enconding for test data.

    This is similar to how we do validation
    """
    ### BEGIN SOLUTION
    new_col = col + "_" + "mean_enc"
    mean_test = train[target].groupby(train[col]).mean().to_dict()
#     train[new_col] = train[col].map(mean_test)
    test[new_col] = test[col].map(mean_test)
    global_mean = train[target].mean()
#     train[new_col].fillna(global_mean,inplace=True)
    test[new_col].fillna(global_mean,inplace=True)    
    ### END SOLUTION

