
import itertools as it
import math

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error

from xgboost.sklearn import XGBRegressor


def round_to_n_sigfig(x, n=3):
    if x == 0:
        return int(x)
    if n < 1:
        n = 1

    x_round = round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))
    if x_round > 10 ** (n - 1):
        x_round = int(x_round)
    return x_round


def recursive_fit(X_train_comb, y_train_comb, X_test_comb, y_test_comb, parameter_dict, use_gridsearchcv=False,
                  target_log=False):

    feature_columns_full = X_train_comb[0].columns.to_list()

    feature_columns = list()
    feature_columns.append(feature_columns_full.copy())
    feature_columns.append(feature_columns_full.copy())

    num_columns_orig = len(feature_columns_full)
    print('Starting number of feature columns: {}\n'.format(num_columns_orig))

    training_results_cols_prefix = ["RMSE_train_", "RMSE_test_", "MAE_test_", "num_features_", "feature_list_",
                                    "feat_high_import_name_", "feat_high_import_val_",
                                    "features_to_remove_"]
    training_results_cols = []
    for ii in range(1, len(X_train_comb)+1):
        training_results_cols.extend([x + str(ii) for x in training_results_cols_prefix])
    training_results_df = pd.DataFrame(columns=training_results_cols)

    hyperparams_list = list(parameter_dict.keys())
    hyperparams_df = pd.DataFrame(columns=hyperparams_list)

    feature_importance_dict_list = []
    for ii in range(0, len(X_train_comb)):
        feature_importance_dict = {}
        for col in feature_columns_full:
            feature_importance_dict[col] = []
        feature_importance_dict_list.append(feature_importance_dict.copy())

    for jj in range(num_columns_orig):

        # ---
        # As the number of features is reduced, perform hyperparameter search to find the best hyperparameters
        # if jj % round(num_columns_orig / 5.) == 0:
        if jj % 15 == 0:

            best_score = None

            for data_jj in range(2):

                if use_gridsearchcv:
                    # Hyperparameter search using GridSearchCV:
                    xgb_reg = XGBRegressor(n_estimators=1000, early_stopping_rounds=20, random_state=42)

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
                    for parameter_dict_tmp in iter(ParameterGrid(parameter_dict)):

                        xgb_reg = XGBRegressor(n_estimators=1000, early_stopping_rounds=20, random_state=42,
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

        out_row = []

        for data_jj in range(2):

            # XGBoost Training:
            xgb_reg = XGBRegressor(n_estimators=1000, early_stopping_rounds=20, random_state=42, **best_params_dict)

            xgb_reg.fit(X_train_comb[data_jj][feature_columns[data_jj]], y_train_comb[data_jj],
                        eval_set=[(X_test_comb[data_jj][feature_columns[data_jj]], y_test_comb[data_jj])], verbose=False)

            y_train_pred = xgb_reg.predict(X_train_comb[data_jj][feature_columns[data_jj]])
            y_test_pred = xgb_reg.predict(X_test_comb[data_jj][feature_columns[data_jj]])

            # TODO: Instead of rounding, go by significant digits [# of digits to be user-configurable]
            train_err = round_to_n_sigfig(mean_squared_error(y_train_comb[data_jj], y_train_pred, squared=False), 5)
            test_err = round_to_n_sigfig(mean_squared_error(y_test_comb[data_jj], y_test_pred, squared=False), 5)

            if target_log:
                test_mae = round_to_n_sigfig(
                    mean_absolute_error(np.expm1(y_test_comb[data_jj]), np.expm1(y_test_pred)), 5)
            else:
                test_mae = round_to_n_sigfig(mean_absolute_error(y_test_comb[data_jj], y_test_pred), 5)

            out_row.extend(
                [train_err, test_err, test_mae, len(feature_columns[data_jj]), ', '.join(feature_columns[data_jj])])

            max_feat_import_ind = np.argmax(xgb_reg.feature_importances_)
            out_row.extend([feature_columns[data_jj][max_feat_import_ind],
                            round(xgb_reg.feature_importances_[max_feat_import_ind], 2)])

            for ii, col in enumerate(feature_columns[data_jj]):
                feature_importance_dict_list[data_jj][col].append(xgb_reg.feature_importances_[ii])

            xx = np.where(xgb_reg.feature_importances_ == 0)[0]
            if xx.size > 0:
                cols_zero_feat_import = [feature_columns[data_jj][x] for x in xx]
                for col in cols_zero_feat_import:
                    feature_columns[data_jj].remove(col)
                col_to_drop = ', '.join(cols_zero_feat_import)
            else:
                min_feat_import_ind = np.argmin(xgb_reg.feature_importances_)
                col_to_drop = feature_columns[data_jj][min_feat_import_ind]
                feature_columns[data_jj].remove(col_to_drop)

            out_row.append(col_to_drop)

        training_results_df.loc[jj] = out_row
        print('Iter', jj, training_results_df.loc[jj, "num_features_1"], training_results_df.loc[jj, "RMSE_test_1"],
              training_results_df.loc[jj, "feat_high_import_name_1"],
              training_results_df.loc[jj, "feat_high_import_val_1"])

        if len(feature_columns[0]) == 0 or len(feature_columns[1]) == 0:
            break

    print()

    return training_results_df, hyperparams_df, feature_importance_dict_list

