
from datetime import datetime
from pathlib import Path

from ._create_pdf_report import (
    initialize_pdf_doc,
    section_on_null_columns,
    section_on_unique_values,
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

    def __init__(self, report_prefix='FeatureSelection', target_col=None,
                 numeric_uniq_vals_thresh=10, nonnumeric_uniq_vals_thresh=5):

        self.report_prefix = report_prefix
        self.target_col = target_col
        self.numeric_uniq_vals_thresh = numeric_uniq_vals_thresh
        self.nonnumeric_uniq_vals_thresh = nonnumeric_uniq_vals_thresh

        self.pdf = None
        self.null_cols_df = None
        self.numeric_cols = None
        self.non_numeric_cols = None
        self.numeric_uniq_vals_df = None
        self.non_numeric_uniq_vals_df = None
        self.numeric_df = None
        self.numeric_collinear_df = None
        self.numeric_collinear_summary_df = None
        self.non_numeric_df = None

    def run_full_eda(self, data_df):

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plots_folder = './{}_EDA_plots_{}'.format(self.report_prefix, timestamp)
        Path(plots_folder).mkdir()

        self.null_cols_df = count_null_values(data_df)

        self.numeric_cols, self.non_numeric_cols = sort_numeric_nonnumeric_columns(data_df, self.target_col)

        self.numeric_uniq_vals_df = count_numeric_unique_values(data_df, self.numeric_cols,
                                                                uniq_vals_thresh=self.numeric_uniq_vals_thresh)

        self.non_numeric_uniq_vals_df = count_nonnumeric_unique_values(data_df, self.non_numeric_cols,
                                                                       uniq_vals_thresh=self.nonnumeric_uniq_vals_thresh)

        # ---
        # TODO: Add feature correlations

        self.numeric_df = calc_numeric_features_target_corr(data_df, self.numeric_cols, self.target_col,
                                                            rf_n_estimators=10)

        self.numeric_collinear_df, self.numeric_collinear_summary_df = calc_corr_numeric_features(data_df,
                                                                                                  self.numeric_cols)

        self.non_numeric_df = calc_nonnumeric_features_target_corr(data_df, self.non_numeric_cols)

        # ---
        # Generating PDF Document
        self.pdf = initialize_pdf_doc()

        # PDF Page 1: Summary of Null values information and unique values for numeric and non-numeric feature columns
        self.pdf = section_on_null_columns(self.pdf, data_df.shape[1], self.null_cols_df)

        self.pdf = section_on_unique_values(self.pdf, self.numeric_cols, self.non_numeric_cols,
                                            self.numeric_uniq_vals_df, self.non_numeric_uniq_vals_df)

        # PDF Pages 2-3: Summary of numeric and non-numeric feature correlations
        self.pdf = section_on_feature_corr(self.pdf, self.numeric_df, self.numeric_collinear_df, self.non_numeric_df)

        # ---
        # TODO: Add plots

        plot_feature_values(data_df, self.numeric_cols, self.numeric_df, target_col=self.target_col, numeric=True,
                            plots_folder=plots_folder)

        self.pdf = section_of_plots(self.pdf, self.numeric_cols, target_col=self.target_col, numeric=True,
                                    plots_folder=plots_folder)


        plot_feature_values(data_df, self.non_numeric_cols, self.non_numeric_df, target_col=self.target_col,
                            numeric=False, plots_folder=plots_folder)

        self.pdf = section_of_plots(self.pdf, self.non_numeric_cols, target_col=self.target_col, numeric=False,
                                    plots_folder=plots_folder)

        # Save PDF document to current working directory
        save_pdf_doc(self.pdf, custom_filename=self.report_prefix, timestamp=timestamp)

