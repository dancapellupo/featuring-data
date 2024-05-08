
import math

from tqdm.auto import tqdm

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    log_loss,
    cohen_kappa_score,
)

from xgboost.sklearn import XGBRegressor, XGBClassifier


def get_metric_names(target_type='regression'):
    if target_type == 'regression':
        primary_metric = 'RMSE'
        secondary_metric = 'MAE'
    else:
        primary_metric = 'logloss'
        secondary_metric = 'CohKap'

    return primary_metric, secondary_metric


def round_to_n_sigfig(x, n=3):
    """
    Round a number to 'n' significant digits.

    Parameters
    ----------
    x : int or float
        Any number to round.

    n : int
        Number of desired significant digits.

    Returns
    -------
    x_round : float or int
        The rounded number.

    Examples
    --------
    >>> round_to_n_sigfig(234.5, n=3)
    235
    >>> round_to_n_sigfig(0.2345, n=3)
    0.235
    """

    # First check if zero is passed to the function to avoid an error:
    if x == 0:
        return int(x)
    # Since n should be at least 1:
    if n < 1:
        n = 1

    # This one line does the actual rounding:
    x_round = round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))
    # If rounding creates a number with no digits beyond the decimal point,
    #  then make it an integer:
    if x_round > 10 ** (n - 1):
        x_round = int(x_round)
    return x_round


def calc_model_metric(y, y_pred, target_type='regression', metric_type='regular'):
        if target_type == 'regression':
            if metric_type == 'regular':
                return mean_squared_error(y, y_pred, squared=False)
            else:
                return mean_absolute_error(y, y_pred)
        else:
            if metric_type == 'regular':
                return log_loss(y, y_pred)
            else:
                return cohen_kappa_score(y, y_pred)


def recursive_fit(X_train_comb, y_train_comb, X_test_comb, y_test_comb, parameter_dict, target_type='regression',
                  use_gridsearchcv=False, target_log=False):
    """
    This is the core function that performs the iterative model training.

    Parameters
    ----------
    X_train_comb : list
        A list of X_train training sets.

    y_train_comb : list
        A list of y_train target values.

    X_test_comb : list
        A list of validation data splits.

    y_test_comb : list
        A list of validation target value splits.

    parameter_dict : dict
        A dictionary of hyperparameters for performing hyperparameter tuning
        for the ML algorithm.

    use_gridsearchcv : bool, default=False
        Whether to use scikit-learn's grid search implementation.

    target_log : bool, default=False
        Whether the target values are the log of the original values.

    Returns
    -------
    training_results_df : pd.DataFrame
        A dataframe with comprehensive results of the iterative model
        training run.
        The index of the dataframe is the number of the iteration,
        starting from iteration 0 with all features included. The
        following columns are generated for each random data split:
        - "RMSE_train_":
        - "RMSE_test_":
        - "MAE_test_":
        - "num_features_":
        - "feature_list_":
        - "feat_high_import_name_":
        - "feat_high_import_val_":
        - "features_to_remove_":

    hyperparams_df : pd.DataFrame
        The best hyperparameters at each iteration where hyperparameter search
        is performed.

    feature_importance_dict_list : list
        A list of dictionaries, with each list corresponding to a different
        split of the data. The dictionaries contain a list of feature
        importance values for each feature, corresponding to the iterations in
        which each feature appears. In other words, if a feature appeared in
        only the first iteration, then the list for that features contains
        just 1 feature importance values. If the feature appeared in the first
        50 iterations before being removed, then its list would contain 50
        feature importance values.
    """

    # ------------------------------------------------------------------------
    feature_columns_full = X_train_comb[0].columns.to_list()

    # Prepare separate copies of the feature lists for each split of the data:
    feature_columns = list()
    feature_columns.append(feature_columns_full.copy())
    feature_columns.append(feature_columns_full.copy())

    num_columns_orig = len(feature_columns_full)
    print('Starting number of feature columns: {}\n'.format(num_columns_orig))

    # Set-up training results dataframe:
    primary_metric, secondary_metric = get_metric_names(target_type)
    training_results_cols_prefix = [
        f"{primary_metric}_train_", f"{primary_metric}_test_", f"{secondary_metric}_test_", "num_features_",
        "feature_list_", "feat_high_import_name_", "feat_high_import_val_", "features_to_remove_"]
    training_results_cols = []
    for ii in range(1, len(X_train_comb)+1):
        training_results_cols.extend([x + str(ii) for x in training_results_cols_prefix])
    training_results_df = pd.DataFrame(columns=training_results_cols)

    # Set-up dataframe to store results of hyperparameter search:
    hyperparams_list = list(parameter_dict.keys())
    hyperparams_df = pd.DataFrame(columns=hyperparams_list)

    # Set-up list of dictionaries to store all the feature importance values
    #  for every iteration of the model training:
    feature_importance_dict_list = []
    for ii in range(0, len(X_train_comb)):
        feature_importance_dict = {}
        for col in feature_columns_full:
            feature_importance_dict[col] = []
        feature_importance_dict_list.append(feature_importance_dict.copy())

    # ------------------------------------------------------------------------
    # Start the Iterative Model Training
    for jj in range(num_columns_orig):

        # As the number of features is reduced, perform hyperparameter search
        #  to find the best hyperparameters:
        # if jj % round(num_columns_orig / 5.) == 0:
        if jj % 15 == 0:

            # ----------------------------------------------------------------
            # Hyperparameter Search
            best_score = None

            for data_jj in range(2):

                if use_gridsearchcv:
                    # Hyperparameter search using GridSearchCV:
                    if target_type == 'regression':
                        xgb_reg = XGBRegressor(n_estimators=1000, early_stopping_rounds=20, random_state=42)
                    else:
                        xgb_reg = XGBClassifier(n_estimators=1000, early_stopping_rounds=20, random_state=42)

                    grid_search = GridSearchCV(xgb_reg, param_grid=parameter_dict, cv=2)

                    grid_search.fit(X_train_comb[data_jj][feature_columns[data_jj]], y_train_comb[data_jj],
                                    eval_set=[(X_test_comb[data_jj][feature_columns[data_jj]], y_test_comb[data_jj])],
                                    verbose=False)

                    if data_jj == 0:
                        best_params_dict = grid_search.best_params_
                        best_score = grid_search.best_score_

                    elif grid_search.best_score_ < best_score:
                        best_params_dict = grid_search.best_params_
                        best_score = grid_search.best_score_

                else:
                    # Hyperparameter search using the train and validation sets already defined:
                    print('Running grid search at Iteration {} on data split {}...'.format(jj, data_jj+1))
                    for parameter_dict_tmp in iter(tqdm(ParameterGrid(parameter_dict))):
                        
                        if target_type == 'regression':
                            xgb_reg = XGBRegressor(n_estimators=1000, early_stopping_rounds=20, random_state=42,
                                                   **parameter_dict_tmp)
                        else:
                            xgb_reg = XGBClassifier(n_estimators=1000, early_stopping_rounds=20, random_state=42,
                                                    **parameter_dict_tmp)
                        xgb_reg.fit(X_train_comb[data_jj][feature_columns[data_jj]], y_train_comb[data_jj],
                                    eval_set=[(X_test_comb[data_jj][feature_columns[data_jj]], y_test_comb[data_jj])],
                                    verbose=False)

                        if (best_score is None) or (xgb_reg.best_score < best_score):
                            best_score = xgb_reg.best_score
                            best_params_dict = parameter_dict_tmp

            out_row = []
            for hyperparam in hyperparams_list:
                out_row.append(best_params_dict[hyperparam])
            hyperparams_df.loc[jj] = out_row
            print('\nIter {} -- New best params: {}\n'.format(jj, best_params_dict))

        # --------------------------------------------------------------------
        # Iterative Model Training

        out_row = []

        # Loop over the two random data splits:
        for data_jj in range(2):

            # XGBoost Training:
            if target_type == 'regression':
                xgb_reg = XGBRegressor(n_estimators=1000, early_stopping_rounds=20, random_state=42, **best_params_dict)
            else:
                xgb_reg = XGBClassifier(n_estimators=1000, early_stopping_rounds=20, random_state=42, **best_params_dict)

            xgb_reg.fit(X_train_comb[data_jj][feature_columns[data_jj]], y_train_comb[data_jj],
                        eval_set=[(X_test_comb[data_jj][feature_columns[data_jj]], y_test_comb[data_jj])], verbose=False)

            if target_type == 'regression':
                y_train_pred = xgb_reg.predict(X_train_comb[data_jj][feature_columns[data_jj]])
                y_test_pred = xgb_reg.predict(X_test_comb[data_jj][feature_columns[data_jj]])
            else:
                y_train_pred = xgb_reg.predict_proba(X_train_comb[data_jj][feature_columns[data_jj]])
                y_test_pred = xgb_reg.predict_proba(X_test_comb[data_jj][feature_columns[data_jj]])

            # TODO: Instead of rounding, go by significant digits [# of digits to be user-configurable]
            train_err = round_to_n_sigfig(calc_model_metric(y_train_comb[data_jj], y_train_pred, target_type=target_type), 5)
            test_err = round_to_n_sigfig(calc_model_metric(y_test_comb[data_jj], y_test_pred, target_type=target_type), 5)

            # If the log of the training data was taken, then reverse the log
            #  to save an easier-to-follow MAE value for the user:
            if target_log:
                test_mae = round_to_n_sigfig(
                    calc_model_metric(np.expm1(y_test_comb[data_jj]), np.expm1(y_test_pred), target_type=target_type, metric_type='easy'), 5)
            else:
                if target_type == 'classification':
                    y_test_pred = xgb_reg.predict(X_test_comb[data_jj][feature_columns[data_jj]])
                test_mae = round_to_n_sigfig(calc_model_metric(y_test_comb[data_jj], y_test_pred, target_type=target_type, metric_type='easy'), 5)
            
            # ----------------------------------------------------------------
            # Save information from this iteration to dataframe
            out_row.extend(
                [train_err, test_err, test_mae, len(feature_columns[data_jj]), ', '.join(feature_columns[data_jj])])

            max_feat_import_ind = np.argmax(xgb_reg.feature_importances_)
            out_row.extend([feature_columns[data_jj][max_feat_import_ind],
                            round(xgb_reg.feature_importances_[max_feat_import_ind], 2)])

            # Save the feature importance values to the list of dictionaries:
            for ii, col in enumerate(feature_columns[data_jj]):
                feature_importance_dict_list[data_jj][col].append(xgb_reg.feature_importances_[ii])

            # ----------------------------------------------------------------
            # Determine which Features to Remove this Iteration

            # First check if there are multiple features with an importance of
            #  exactly zero:
            xx = np.where(xgb_reg.feature_importances_ == 0)[0]
            if xx.size > 0:
                cols_zero_feat_import = [feature_columns[data_jj][x] for x in xx]
                # Remove all features with an importance of exactly zero, if
                #  there are any:
                for col in cols_zero_feat_import:
                    feature_columns[data_jj].remove(col)
                col_to_drop = ', '.join(cols_zero_feat_import)
            else:
                # In most cases, just remove the feature with the lowest, but
                #  non-zero feature importance:
                min_feat_import_ind = np.argmin(xgb_reg.feature_importances_)
                col_to_drop = feature_columns[data_jj][min_feat_import_ind]
                feature_columns[data_jj].remove(col_to_drop)

            # Save to dataframe the name(s) of the dropped column(s):
            out_row.append(col_to_drop)

        training_results_df.loc[jj] = out_row
        print('Iter', jj, training_results_df.loc[jj, "num_features_1"],
              training_results_df.loc[jj, f"{primary_metric}_test_1"],
              training_results_df.loc[jj, "feat_high_import_name_1"],
              training_results_df.loc[jj, "feat_high_import_val_1"])

        # Stop running the iterative training once all features have been
        #  removed from at least one of the data splits:
        if len(feature_columns[0]) == 0 or len(feature_columns[1]) == 0:
            break

    print()

    return training_results_df, hyperparams_df, feature_importance_dict_list

