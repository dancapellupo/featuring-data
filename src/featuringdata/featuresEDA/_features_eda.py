
from ._create_pdf_report import (
    initialize_pdf_doc,
    section_on_null_columns,
    section_on_unique_values,
    save_pdf_doc
)

from ._initial_eda_functions import count_null_values, sort_numeric_nonnumeric_columns
from ._initial_eda_functions import count_numeric_unique_values, count_nonnumeric_unique_values


class FeaturesEDA:

    def __init__(self, target_col=None,
                 numeric_uniq_vals_thresh=10, nonnumeric_uniq_vals_thresh=5):

        self.target_col = target_col
        self.numeric_uniq_vals_thresh = numeric_uniq_vals_thresh
        self.nonnumeric_uniq_vals_thresh = nonnumeric_uniq_vals_thresh

        self.pdf = None
        self.null_cols_df = None
        self.numeric_cols = None
        self.non_numeric_cols = None
        self.numeric_uniq_vals_df = None
        self.non_numeric_uniq_vals_df = None

    def run_full_eda(self, data_df):

        self.null_cols_df = count_null_values(data_df)

        self.numeric_cols, self.non_numeric_cols = sort_numeric_nonnumeric_columns(data_df, self.target_col)

        self.numeric_uniq_vals_df = count_numeric_unique_values(data_df, self.numeric_cols,
                                                                uniq_vals_thresh=self.numeric_uniq_vals_thresh)

        self.non_numeric_uniq_vals_df = count_nonnumeric_unique_values(data_df, self.non_numeric_cols,
                                                                       uniq_vals_thresh=self.nonnumeric_uniq_vals_thresh)

        # ---

        self.pdf = initialize_pdf_doc()

        self.pdf = section_on_null_columns(self.pdf, data_df.shape[1], self.null_cols_df)

        self.pdf = section_on_unique_values(self.pdf, self.numeric_cols, self.non_numeric_cols,
                                            self.numeric_uniq_vals_df, self.non_numeric_uniq_vals_df)

        save_pdf_doc(self.pdf)

