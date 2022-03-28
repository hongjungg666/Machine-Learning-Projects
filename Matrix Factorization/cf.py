import numpy as np
import pandas as pd
from scipy import sparse

def proc_col(col):
    """Encodes a pandas column with values between 0 and n-1.
     as the order of the first time a index appear
    where n = number of unique values
    """
    uniq = col.unique()
    name2idx = {o:i for i,o in enumerate(uniq)}
    return name2idx, np.array([name2idx[x] for x in col]), len(uniq)

def encode_data(df):
    """Encodes rating data with continous user and movie ids using 
    the helpful fast.ai function from above.
    
    Arguments:
      train_csv: a csv file with columns user_id,movie_id,rating 
    
    Returns:
      df: a dataframe with the encode data
      num_users
      num_movies
      
    """
    ### BEGIN SOLUTION
    name2idx,df["userId"],num_users = proc_col(df["userId"])
    name2idx_2,df["movieId"],num_movies = proc_col(df["movieId"])
    ### END SOLUTION
    return df, num_users, num_movies

def encode_new_data(df_val, df_train):
    """ Encodes df_val with the same encoding as df_train.
    Returns:
    df_val: dataframe with the same encoding as df_train
    """
    ### BEGIN SOLUTION
    
    ### END SOLUTION
    return df_val

def create_embedings(n, K):
    """ Create a numpy random matrix of shape n, K
    
    The random matrix should be initialized with uniform values in (0, 6/K)
    Arguments:
    
    Inputs:
    n: number of items/users
    K: number of factors in the embeding 
    
    Returns:
    emb: numpy array of shape (n, num_factors)
    """
    np.random.seed(3)
    emb = 6*np.random.random((n, K)) / K
    return emb


def df2matrix(df, nrows, ncols, column_name="rating"):
    """ Returns a sparse matrix constructed from a dataframe
    
    This code assumes the df has columns: MovieID,UserID,Rating
    """
    values = df[column_name].values
    ind_movie = df['movieId'].values
    ind_user = df['userId'].values
    return sparse.csc_matrix((values,(ind_user, ind_movie)),shape=(nrows, ncols))

def sparse_multiply(df, emb_user, emb_movie):
    """ This function returns U*V^T element wise multi by R as a sparse matrix.
    
    It avoids creating the dense matrix U*V^T
    """
    
    df["Prediction"] = np.sum(emb_user[df["userId"].values]*emb_movie[df["movieId"].values], axis=1)
    return df2matrix(df, emb_user.shape[0], emb_movie.shape[0], column_name="Prediction")

def cost(df, emb_user, emb_movie):
    """ Computes mean square error

    First compute prediction. Prediction for user i and movie j is
    emb_user[i]*emb_movie[j]

    Arguments:
      df: dataframe with all data or a subset of the data
      emb_user: embedings for users
      emb_movie: embedings for movies

    Returns:
      error(float): this is the MSE
    """
    # BEGIN SOLUTION
    prediction = sparse_multiply(df, emb_user, emb_movie).data
    Y = df2matrix(df, nrows=emb_user.shape[0], ncols=emb_movie.shape[0]).data
    error = np.mean((Y - prediction)**2)
    return error

def finite_difference(df, emb_user, emb_movie, ind_u=None, ind_m=None, k=None):
    """ Computes finite difference on MSE(U, V).
    
    This function is used for testing the gradient function. 
    """
    e = 0.000000001
    c1 = cost(df, emb_user, emb_movie)
    K = emb_user.shape[1]
    x = np.zeros_like(emb_user)
    y = np.zeros_like(emb_movie)
    if ind_u is not None:
        x[ind_u][k] = e
    else:
        y[ind_m][k] = e
    c2 = cost(df, emb_user + x, emb_movie + y)
    return (c2 - c1)/e

def gradient(df, Y, emb_user, emb_movie):
    """ Computes the gradient.
    
    First compute prediction. Prediction for user i and movie j is
    emb_user[i]*emb_movie[j]
    
    Arguments:
      df: dataframe with all data or a subset of the data
      Y: sparse representation of df
      emb_user: embedings for users
      emb_movie: embedings for movies
      
    Returns:
      d_emb_user
      d_emb_movie
    """
    ### BEGIN SOLUTION
    N = df.shape[0]
    delta = Y.toarray() - sparse_multiply(df, emb_user, emb_movie).toarray()
    grad_user = (-2.00)/N*np.dot(delta,emb_movie)
    grad_movie = (-2.00)/N*np.dot(delta.transpose(),emb_user)
    ### END SOLUTION
    return grad_user, grad_movie

# you can use a for loop to iterate through gradient descent
def gradient_descent(df, emb_user, emb_movie, iterations=100, learning_rate=0.01, df_val=None):
    """ Computes gradient descent with momentum (0.9) for a number of iterations.

    Prints training cost and validation cost (if df_val is not None) every 50 iterations.

    Returns:
    emb_user: the trained user embedding
    emb_movie: the trained movie embedding
    """
    Y = df2matrix(df, emb_user.shape[0], emb_movie.shape[0])
    # BEGIN SOLUTION
    moment_user = 0
    moment_movie = 0
    for i in range(iterations):
        grad_user, grad_movie = gradient(df, Y, emb_user, emb_movie)
        moment_user = 0.9*moment_user + (1-0.9)*grad_user
        moment_movie = 0.9*moment_movie + (1-0.9)*grad_movie
        emb_user = emb_user-learning_rate*moment_user
        emb_movie = emb_movie-learning_rate*moment_movie
        if (i+1) % 50 == 0:
            if df_val is None:
                cost_train = cost(df, emb_user, emb_movie)
                print(i+1,"training cost:", cost_train," None")
            else:
                cost_train = cost(df, emb_user, emb_movie)
                cost_val = cost(df_val, emb_user, emb_movie)
                print(i+1,"training cost:", cost_train,"validation cost:", cost_val)
    # END SOLUTION
    return emb_user, emb_movie


def proc_col_val(col,df_val,df_train):
    """Encodes a pandas column with values between 0 and n-1.
     as the order of the first time a index appear
    where n = number of unique values
    """
    uniq = df_train[col].unique()
    name2idx = {o:i for i,o in enumerate(uniq)}
    return np.array([name2idx[x] for x in df_val[col]])


def encode_new_data(df_val, df_train):
    """ Encodes df_val with the same encoding as df_train.
    Returns:
    df_val: dataframe with the same encoding as df_train
    """
    ### BEGIN SOLUTION
    #filter out new_items
    df_val = df_val[df_val["userId"].isin(df_train["userId"])]
    df_val = df_val[df_val["movieId"].isin(df_train["movieId"])]
    #encoding
    df_val["userId"] = proc_col_val("userId",df_val,df_train)
    df_val["movieId"] = proc_col_val("movieId",df_val,df_train)
    ### END SOLUTION
    return df_val
