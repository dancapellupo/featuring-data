
import itertools as it

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

from xgboost.sklearn import XGBRegressor


def recursive_fit(X_train_comb, y_train_comb, X_test_comb, y_test_comb, parameter_dict):

    feature_columns_full = X_train_comb[0].columns.to_list()

    feature_columns = list()
    feature_columns.append(feature_columns_full.copy())
    feature_columns.append(feature_columns_full.copy())

    num_columns_orig = len(feature_columns_full)
    print('Starting number of feature columns: {}\n'.format(num_columns_orig))

    training_results_df = pd.DataFrame(columns=["RMSE_train_1", "RMSE_test_1", "num_features_1", "feature_list_1", "features_to_remove_1", "RMSE_train_2", "RMSE_test_2", "num_features_2", "feature_list_2", "features_to_remove_2"])
    # training_results_df = pd.DataFrame(
    #     columns=["RMSE_train_1", "RMSE_test_1", "MAE_test_1", "num_features_1", "feature_list_1",
    #              "features_to_remove_1", "RMSE_train_2", "RMSE_test_2", "MAE_test_2", "num_features_2", "feature_list_2",
    #              "features_to_remove_2"])

    hyperparams_list = list(parameter_dict.keys())
    hyperparams_df = pd.DataFrame(columns=hyperparams_list)

    for jj in range(num_columns_orig):

        # if jj % round(num_columns_orig / 5.) == 0:
        if jj % 15 == 0:

            for data_jj in range(2):
                # GridSearchCV + XGBoost Training:
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

            out_row = []
            for hyperparam in hyperparams_list:
                out_row.append(best_params_dict[hyperparam])
            hyperparams_df.loc[jj] = out_row
            print('\nIter {} -- New best params: {}\n'.format(jj, best_params_dict))

        out_row = []

        for data_jj in range(2):

            # XGBoost Training:
            xgb_reg = XGBRegressor(n_estimators=1000, max_depth=best_params_dict["max_depth"], random_state=42,
                                   early_stopping_rounds=20,
                                   gamma=best_params_dict["gamma"], min_child_weight=best_params_dict["min_child_weight"],
                                   max_delta_step=best_params_dict["max_delta_step"])

            xgb_reg.fit(X_train_comb[data_jj][feature_columns[data_jj]], y_train_comb[data_jj],
                        eval_set=[(X_test_comb[data_jj][feature_columns[data_jj]], y_test_comb[data_jj])], verbose=False)

            y_train_pred = xgb_reg.predict(X_train_comb[data_jj][feature_columns[data_jj]])
            y_test_pred = xgb_reg.predict(X_test_comb[data_jj][feature_columns[data_jj]])

            train_err = round(mean_squared_error(y_train_comb[data_jj], y_train_pred, squared=False), 3)
            test_err = round(mean_squared_error(y_test_comb[data_jj], y_test_pred, squared=False), 3)

            out_row.extend([train_err, test_err, len(feature_columns[data_jj]), ', '.join(feature_columns[data_jj])])

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
        # print(jj, out_row[1], out_row[2], out_row[4], out_row[6], out_row[7], out_row[9])
        print('Iter', jj, out_row[1], out_row[2], out_row[6], out_row[7])

        if len(feature_columns[0]) == 0 or len(feature_columns[1]) == 0:
            break

    print()

    return training_results_df, hyperparams_df

