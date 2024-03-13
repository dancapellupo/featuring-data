
from datetime import datetime
from pathlib import Path

import pandas as pd

from ._create_pdf_report import (
    initialize_pdf_doc,
    section_on_null_columns,
    section_on_unique_values,
    section_on_unique_values_p2,
    section_on_feature_corr,
    section_of_plots,
    save_pdf_doc
)

from ._initial_eda_functions import (
    count_null_values,
    sort_numeric_nonnumeric_columns,
    count_numeric_unique_values,
    count_nonnumeric_unique_values
)

from ._correlation import (
    calc_numeric_features_target_corr,
    calc_corr_numeric_features,
    calc_nonnumeric_features_target_corr
)

from ._generate_plots import plot_feature_values


class FeaturesEDA:

    def __init__(self, report_prefix='FeatureSelection', target_col=None, cols_to_drop=None,
                 numeric_uniq_vals_thresh=10, nonnumeric_uniq_vals_thresh=5):

        self.report_prefix = report_prefix
        self.target_col = target_col
        self.cols_to_drop = cols_to_drop
        self.numeric_uniq_vals_thresh = numeric_uniq_vals_thresh
        self.nonnumeric_uniq_vals_thresh = nonnumeric_uniq_vals_thresh

        self.pdf = None
        self.null_cols_df = None
        self.numeric_cols = None
        self.non_numeric_cols = None
        self.numeric_uniq_vals_df = None
        self.non_numeric_uniq_vals_df = None
        self.numeric_df = None
        self.numeric_collinear_df = pd.DataFrame()
        self.numeric_collinear_summary_df = pd.DataFrame()
        self.non_numeric_df = None

    # TODO Create function that does up to correlation
    # TODO For full EDA, make collinear correlation optional

    def run_initial_eda(self, data_df, output=True):

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # TODO: Add a check for columns with words like "ID" to suggest dropping them
        if self.cols_to_drop is not None:
            data_df = data_df.drop(columns=self.cols_to_drop)

        self.null_cols_df = count_null_values(data_df)

        self.numeric_cols, self.non_numeric_cols = sort_numeric_nonnumeric_columns(data_df, self.target_col)

        self.numeric_uniq_vals_df = count_numeric_unique_values(data_df, self.numeric_cols,
                                                                uniq_vals_thresh=self.numeric_uniq_vals_thresh)

        self.non_numeric_uniq_vals_df = count_nonnumeric_unique_values(data_df, self.non_numeric_cols,
                                                                       uniq_vals_thresh=self.nonnumeric_uniq_vals_thresh)

        single_value_cols_numeric_df = self.numeric_uniq_vals_df.loc[
            self.numeric_uniq_vals_df["Num Unique Values"] == 1]

        numeric_cols_to_cat_df = self.numeric_uniq_vals_df.loc[
            self.numeric_uniq_vals_df["Num Unique Values"].between(2, 2, inclusive='both')]

        single_value_cols_nonnumeric_df = self.non_numeric_uniq_vals_df.loc[
            self.non_numeric_uniq_vals_df["Num Unique Values"] == 1]

        print('There are {} numeric and {} non-numeric columns with only a single value.'.format(
            len(single_value_cols_numeric_df), len(single_value_cols_nonnumeric_df)))

        print('There are {} numeric columns that will be switched to categorical.'.format(len(numeric_cols_to_cat_df)))

        # ---
        # Generating PDF Document
        self.pdf = initialize_pdf_doc()

        # PDF Page 1: Summary of Null values information and unique values for numeric and non-numeric feature columns
        self.pdf = section_on_null_columns(self.pdf, data_df.shape[1], self.null_cols_df)

        self.pdf = section_on_unique_values(self.pdf, self.numeric_cols, self.non_numeric_cols,
                                            self.numeric_uniq_vals_df, self.non_numeric_uniq_vals_df,
                                            single_value_cols_numeric_df=single_value_cols_numeric_df,
                                            single_value_cols_nonnumeric_df=single_value_cols_nonnumeric_df,
                                            numeric_cols_to_cat_df=numeric_cols_to_cat_df)

        if ((len(single_value_cols_numeric_df) > 0) or (len(single_value_cols_nonnumeric_df) > 0) or
                (len(numeric_cols_to_cat_df) > 0)):

            if len(single_value_cols_numeric_df) > 0:
                for col in single_value_cols_numeric_df["Feature"].values:
                    self.numeric_cols.remove(col)

            if len(single_value_cols_nonnumeric_df) > 0:
                for col in single_value_cols_nonnumeric_df["Feature"].values:
                    self.non_numeric_cols.remove(col)

            if len(numeric_cols_to_cat_df) > 0:
                for col in numeric_cols_to_cat_df["Feature"]:
                    self.numeric_cols.remove(col)
                    self.non_numeric_cols.append(col)

            self.pdf = section_on_unique_values_p2(self.pdf, self.numeric_cols, self.non_numeric_cols)

        # Save PDF document to current working directory
        if output:
            custom_filename = self.report_prefix + '_Initial'
            save_pdf_doc(self.pdf, custom_filename=custom_filename, timestamp=timestamp)

    def run_full_eda(self, data_df, run_collinear=True, generate_plots=True):

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plots_folder = './{}_EDA_plots_{}'.format(self.report_prefix, timestamp)
        Path(plots_folder).mkdir()

        # if self.cols_to_drop is not None:
        #     data_df = data_df.drop(columns=self.cols_to_drop)

        self.run_initial_eda(data_df, output=False)

        # ---
        # TODO: Add feature correlations

        self.numeric_df = calc_numeric_features_target_corr(data_df, self.numeric_cols, self.target_col,
                                                            rf_n_estimators=10)

        if run_collinear:
            self.numeric_collinear_df, self.numeric_collinear_summary_df = calc_corr_numeric_features(data_df,
                                                                                                      self.numeric_cols)

        self.non_numeric_df = calc_nonnumeric_features_target_corr(data_df, self.non_numeric_cols, self.target_col)

        # ---
        # Generating PDF Document
        # PDF Page 1: Summary of Null values information and unique values for numeric and non-numeric feature columns

        # PDF Pages 2-3: Summary of numeric and non-numeric feature correlations
        self.pdf = section_on_feature_corr(self.pdf, self.numeric_df, self.numeric_collinear_df, self.non_numeric_df)

        # ---
        # TODO: Add plots

        if generate_plots:
            columns_list_ordered = self.numeric_df.index

            plot_feature_values(data_df, columns_list_ordered, self.numeric_df, target_col=self.target_col,
                                numeric=True, plots_folder=plots_folder)

            self.pdf = section_of_plots(self.pdf, columns_list_ordered, target_col=self.target_col, numeric=True,
                                        plots_folder=plots_folder)


            columns_list_ordered = self.non_numeric_df.index

            plot_feature_values(data_df, columns_list_ordered, self.non_numeric_df, target_col=self.target_col,
                                numeric=False, plots_folder=plots_folder)

            self.pdf = section_of_plots(self.pdf, columns_list_ordered, target_col=self.target_col, numeric=False,
                                        plots_folder=plots_folder)

        # Save PDF document to current working directory
        save_pdf_doc(self.pdf, custom_filename=self.report_prefix, timestamp=timestamp)

