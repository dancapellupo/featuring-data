
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from xgboost.sklearn import XGBRegressor


class FeatureSelector:

    def __init__(self, numeric_cols, non_numeric_cols, report_prefix='FeatureSelection', target_col=None,
                 test_size=0.15):

        self.numeric_cols = numeric_cols
        self.non_numeric_cols = non_numeric_cols
        self.report_prefix = report_prefix
        self.target_col = target_col

        self.test_size = test_size


    def run(self, data_df):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plots_folder = './{}_ModelTraining_plots_{}'.format(self.report_prefix, timestamp)
        # Path(plots_folder).mkdir()

        # all_columns = self.numeric_cols.extend(self.non_numeric_cols)

        X = data_df[self.numeric_cols]

        X_onehot = pd.get_dummies(data_df[self.non_numeric_cols], dtype=int)

        X = X.merge(X_onehot, left_index=True, right_index=True)

        y = data_df[self.target_col].values
        y = np.log1p(data_df[self.target_col].values)

        X_train_42, X_test_42, y_train_42, y_test_42 = train_test_split(X, y, test_size=self.test_size,
                                                                        random_state=42)
        X_train_46, X_test_46, y_train_46, y_test_46 = train_test_split(X, y, test_size=self.test_size,
                                                                        random_state=46)

        # TODO Allow user to set max/min values of these hyperparam ranges, as well as number of total iterations,
        #  which would define how many values to consider per hyperparam
        max_depth_list = [3, 4, 5, 6]
        gamma_list = [0, 1, 5]
        min_child_weight_list = [0, 1, 5]
        max_delta_step_list = [0, 1, 5]

        best_params_dict = {"max_depth": max_depth_list[0], "gamma": gamma_list[0],
                            "min_child_weight": min_child_weight_list[0], "max_delta_step": max_delta_step_list[0]}

        feature_columns = X.columns.to_list()

        feature_columns_2 = list()
        feature_columns_2.append(feature_columns.copy())
        feature_columns_2.append(feature_columns.copy())

        num_columns_orig = len(feature_columns)
        print(num_columns_orig)

        X_train_comb = [X_train_42, X_train_46]
        X_test_comb = [X_test_42, X_test_46]

        y_train_comb = [y_train_42, y_train_46]
        y_test_comb = [y_test_42, y_test_46]

        training_results_df = pd.DataFrame(columns=["RMSE_train_1", "RMSE_test_1", "num_features_1", "feature_list_1", "features_to_remove_1", "RMSE_train_2", "RMSE_test_2", "num_features_2", "feature_list_2", "features_to_remove_2"])

        for jj in range(num_columns_orig):

            # if jj % round(num_columns_orig / 5.) == 0:
            if jj % 15 == 0:
                # if 0:

                best_iterations = []
                best_scores = []

                best_score = 1e10

                for max_depth in max_depth_list:

                    for gamma in gamma_list:

                        for min_child_weight in min_child_weight_list:

                            for max_delta_step in max_delta_step_list:

                                for data_jj in range(2):

                                    # XGBoost Training:
                                    xgb_reg = XGBRegressor(n_estimators=1000, max_depth=max_depth, random_state=42,
                                                           early_stopping_rounds=20,
                                                           gamma=gamma, min_child_weight=min_child_weight, max_delta_step=max_delta_step)

                                    xgb_reg.fit(X_train_comb[data_jj][feature_columns_2[data_jj]], y_train_comb[data_jj],
                                                eval_set=[(X_test_comb[data_jj][feature_columns_2[data_jj]], y_test_comb[data_jj])], verbose=False)

                                    best_iterations.append(xgb_reg.best_iteration)
                                    best_scores.append(xgb_reg.best_score)

                                    if xgb_reg.best_score < best_score:
                                        best_params_dict = {"max_depth": max_depth, "gamma": gamma,
                                                            "min_child_weight": min_child_weight, "max_delta_step": max_delta_step}
                                        best_score = xgb_reg.best_score

                print(jj, best_params_dict)

            out_row = []

            for data_jj in range(2):

                # XGBoost Training:
                xgb_reg = XGBRegressor(n_estimators=1000, max_depth=best_params_dict["max_depth"], random_state=42,
                                       early_stopping_rounds=20,
                                       gamma=best_params_dict["gamma"], min_child_weight=best_params_dict["min_child_weight"],
                                       max_delta_step=best_params_dict["max_delta_step"])

                xgb_reg.fit(X_train_comb[data_jj][feature_columns_2[data_jj]], y_train_comb[data_jj],
                            eval_set=[(X_test_comb[data_jj][feature_columns_2[data_jj]], y_test_comb[data_jj])], verbose=False)

                y_train_pred = xgb_reg.predict(X_train_comb[data_jj][feature_columns_2[data_jj]])
                y_test_pred = xgb_reg.predict(X_test_comb[data_jj][feature_columns_2[data_jj]])

                train_err = round(mean_squared_error(y_train_comb[data_jj], y_train_pred, squared=False), 3)
                test_err = round(mean_squared_error(y_test_comb[data_jj], y_test_pred, squared=False), 3)

                out_row.extend([train_err, test_err, len(feature_columns_2[data_jj]), ', '.join(feature_columns_2[data_jj])])

                xx = np.where(xgb_reg.feature_importances_ == 0)[0]
                if xx.size > 0:
                    cols_zero_feat_import = [feature_columns_2[data_jj][x] for x in xx]
                    for col in cols_zero_feat_import:
                        feature_columns_2[data_jj].remove(col)
                    col_to_drop = ', '.join(cols_zero_feat_import)
                else:
                    min_feat_import_ind = np.argmin(xgb_reg.feature_importances_)
                    col_to_drop = feature_columns_2[data_jj][min_feat_import_ind]
                    feature_columns_2[data_jj].remove(col_to_drop)

                out_row.append(col_to_drop)

            training_results_df.loc[jj] = out_row
            # print(jj, out_row[1], out_row[2], out_row[4], out_row[6], out_row[7], out_row[9])
            print(jj, out_row[1], out_row[2], out_row[6], out_row[7])

            if len(feature_columns_2[0]) == 0 or len(feature_columns_2[1]) == 0:
                break

        print(np.argmin(training_results_df["RMSE_test_1"].values))
        print(np.argmin(training_results_df["RMSE_test_2"].values))

        X_train_best = X_train_comb[1][training_results_df.loc[61, "feature_list_2"].split(', ')]
        X_test_best = X_test_comb[1][training_results_df.loc[61, "feature_list_2"].split(', ')]

        # XGBoost Training:
        xgb_reg = XGBRegressor(n_estimators=1000, max_depth=3, early_stopping_rounds=20, random_state=42, gamma=0)
        xgb_reg.fit(X_train_best, y_train_comb[1], eval_set=[(X_test_best, y_test_comb[1])],
                    verbose=True)

        y_test_pred = xgb_reg.predict(X_test_best)
        print(mean_absolute_error(np.expm1(y_test_46), np.expm1(y_test_pred)))

        return training_results_df



