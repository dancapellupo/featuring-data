
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from xgboost.sklearn import XGBRegressor

from ._recursive_fit import recursive_fit


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

        X_train_comb = [X_train_42, X_train_46]
        X_test_comb = [X_test_42, X_test_46]

        y_train_comb = [y_train_42, y_train_46]
        y_test_comb = [y_test_42, y_test_46]

        training_results_df = recursive_fit(X_train_comb, y_train_comb, X_test_comb, y_test_comb)

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

        training_results_df.to_csv('{}_training_results_full_{}.csv'.format(self.report_prefix, timestamp))

        return training_results_df



