"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).

    Args:
        X: (N, F) array-like of floats, may contain NaN.

    Returns:
        (N, F) float ndarray with no NaNs.
    """
    # TODO: Replace every NaN with that column's nan-aware mean...
    rows,cols= np.shape(X)
    for col in range(cols):
        nans= np.sum(np.isnan(X[:,col]))
        if nans==rows:
             np.nan_to_num(X[:,col], nan=0., copy=False)
        else:
            mean= np.nanmean(X[:,col])
            np.nan_to_num(X[:,col], nan=mean, copy=False)
    return X

# Step 2 - compute_iqr_bounds
import numpy as np
def compute_iqr_bounds(X, k=1.5):
    # TODO: Compute per-column lower/upper clip bounds using the IQR rule.
    q1= 25
    q3= 75
    f= np.shape(X)[1]
    lower= np.zeros((f,))
    upper= np.zeros((f,))
    np.percentile(X,q=q1,out=lower,axis=0)
    np.percentile(X,q=q3,out=upper,axis=0)
    iqr= upper-lower
    lower= lower- k*iqr
    upper= upper+ k*iqr
    return lower,upper

# Step 3 - clip_columns
def clip_columns(X, lower, upper):
    # TODO: Clip every entry of a feature matrix to per-column lower/upper bounds.
    out= np.zeros(np.shape(X))
    np.clip(X, lower, upper, out=out)
    return out

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    # TODO: Form a derived ratio feature from two 1-D arrays with safe division.
    return numerator/(denominator+eps)

# Step 5 - append_column
def append_column(X, col):
    # TODO: Horizontally append one 1-D feature column onto a design matrix.
   
    return np.concatenate([X,col.reshape(-1,1)], axis=1)

# Step 6 - one_hot_encode
def one_hot_encode(labels):
    # TODO: Convert a 1-D array of categorical labels into a dense binary one-hot matrix.
    uniq=np.sort(np.unique(labels))
    out=np.zeros((np.size(labels),np.size(uniq)))
    for i,label in enumerate(labels):
        out[i,:]= np.float64(uniq==label)
    return out

# Step 7 - fit_standardizer
def fit_standardizer(X):
    # TODO: Compute per-column mean and std used to standardize features...
    mean=np.mean(X,axis=0)
    std=np.std(X,axis=0)
    std=np.where(std==0.,1.,std)
    return mean , std

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    # TODO: Return the scaled matrix (X - mean) / std via broadcasting.
    return (X-mean)/std

# Step 9 - add_bias_column
def add_bias_column(X):
    # TODO: Prepend a column of ones to a 2-D feature matrix X...
    return np.concatenate([np.ones((X.shape[0],1)), X], axis=1)

# Step 10 - make_shuffled_indices
def make_shuffled_indices(n_samples, seed):
    # TODO: Create a reproducibly shuffled permutation of row indices.
    np.random.seed(seed)
    return np.random.permutation(n_samples)

# Step 11 - partition_indices
def partition_indices(indices, train_ratio, val_ratio):
    # TODO: Split a shuffled index array into train, validation, and test index arrays.
    n= np.size(indices)
    train = indices[:int(n*train_ratio)]
    val= indices[int(n*train_ratio):int(n*train_ratio+n*val_ratio)]
    test=indices[int(n*train_ratio+n*val_ratio):]
    return train, val,test

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    # TODO: Select the rows of X and y at the given indices.
    xsub= X[indices,:]
    ysub=y[indices]
    return xsub,ysub

# Step 13 - ols_fit
def ols_fit(X, y):
    # TODO: return the ordinary-least-squares weight vector for a linear model.
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    s_inv = np.where(s != 0, 1 / s, 0)
    ps = vt.T @ np.diag(s_inv) @ u.T
    return ps @ y

# Step 14 - ols_predict
def ols_predict(X, theta):
    # TODO: Predict continuous targets with a fitted linear model.
    return X@ theta

# Step 15 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    # TODO: return the mean absolute error between targets and predictions
    return (1/np.size(y_pred)) * np.sum(np.abs(y_pred-y_true))

# Step 16 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    """Compute root mean squared error between targets and predictions.

    Args:
        y_true (np.ndarray): Ground-truth targets, shape (N,).
        y_pred (np.ndarray): Predicted targets, shape (N,).

    Returns:
        float: RMSE value.
    """
    # TODO: return the root mean squared error as a Python float
    return np.sqrt(
        (1/np.size(y_pred))* np.sum(
            (y_pred-y_true)**2
        )
    )

# Step 17 - r_squared
def r_squared(y_true, y_pred):
    # TODO: Compute R^2 = 1 - SS_res/SS_tot (return 0.0 if SS_tot is 0)...
    ssr= np.sum((y_true-y_pred)**2)
    mean=np.mean(y_true,keepdims=True)
    sst=np.sum((y_true-mean)**2)
    return 1-(ssr/sst) if sst!=0.0 else 0

# Step 18 - residual_summary
def residual_summary(y_true, y_pred):
    # TODO: Return a compact dict summarizing prediction residuals...
    r= y_true-y_pred
    return {
        "mean": np.mean(r),
        "std": np.std(r),
        "median_abs": np.median(np.abs(r))
    }

# Step 19 - prepare_cleaned_features (not yet solved)
# TODO: implement

# Step 20 - assemble_feature_matrix (not yet solved)
# TODO: implement

# Step 21 - make_train_val_test (not yet solved)
# TODO: implement

# Step 22 - standardize_and_add_bias (not yet solved)
# TODO: implement

# Step 23 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 24 - house_price_pipeline (not yet solved)
# TODO: implement

