
from fpdf import FPDF


def initialize_pdf_doc():
    pdf = FPDF()

    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(w=0, h=10, txt="Feature Selection and EDA Report", ln=1)
    pdf.ln(2)

    return pdf


def save_pdf_doc(pdf, custom_filename='FeatureSelection', timestamp=''):
    pdf.output('./{}_EDA_Report_{}.pdf'.format(custom_filename, timestamp), 'F')


def section_on_null_columns(pdf, num_features, null_cols_df):
    pdf.set_font('Arial', 'B', 13)
    pdf.cell(w=0, h=10, txt="Null Columns", ln=1)

    pdf.set_font('Arial', '', 12)
    pdf.cell(w=0, h=10,
             txt="Out of {} total feature columns, there are {} columns with at least 1 null value.".format(num_features, len(null_cols_df)),
             ln=1)

    pdf.ln(2)

    # Table Header
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(w=60, h=10, txt=null_cols_df.columns[0], border=1, ln=0, align='C')
    pdf.cell(w=35, h=10, txt=null_cols_df.columns[1], border=1, ln=0, align='C')
    pdf.cell(w=35, h=10, txt=null_cols_df.columns[2], border=1, ln=1, align='C')

    # Table contents
    pdf.set_font('Arial', '', 12)
    for ii in range(0, 5):
        pdf.cell(w=60, h=10, txt=null_cols_df["Feature"].iloc[ii], border=1, ln=0, align='L')
        pdf.cell(w=35, h=10, txt=null_cols_df["Num of Nulls"].iloc[ii].astype(str), border=1, ln=0, align='R')
        pdf.cell(w=35, h=10, txt=null_cols_df["Frac Null"].iloc[ii].astype(str), border=1, ln=1, align='R')

    return pdf


def section_on_unique_values(pdf, numeric_cols, non_numeric_cols, numeric_uniq_vals_df, non_numeric_uniq_vals_df):
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 13)
    pdf.cell(w=0, h=10, txt="Numeric vs Non-Numeric Features", ln=1)

    pdf.set_font('Arial', '', 12)
    pdf.cell(w=0, h=10,
             txt="Out of {} total feature columns, there are {} numeric columns and {} non-numeric columns.".format(
                 len(numeric_cols)+len(non_numeric_cols), len(numeric_cols), len(non_numeric_cols)),
             ln=1)

    pdf.ln(3)

    # Table Header
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(w=60, h=10, txt='Numeric Feature', border=1, ln=0, align='C')
    pdf.cell(w=42, h=10, txt=numeric_uniq_vals_df.columns[1], border=1, ln=1, align='C')

    # Table Contents
    pdf.set_font('Arial', '', 12)
    for ii in range(0, min(5, len(numeric_uniq_vals_df))):
        pdf.cell(w=60, h=10,
                 txt=numeric_uniq_vals_df["Feature"].iloc[ii],
                 border=1, ln=0, align='L')
        pdf.cell(w=42, h=10,
                 txt=numeric_uniq_vals_df["Num Unique Values"].iloc[ii].astype(str),
                 border=1, ln=1, align='R')

    if len(numeric_uniq_vals_df) > 5:
        pdf.cell(w=0, h=10,
                 txt="There are an additional {} numeric feature columns with 10 or fewer unique values.".format(
                     len(numeric_uniq_vals_df) - 5),
                 ln=1)

    pdf.ln(4)

    # Table Header
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(w=60, h=10, txt='Non-Numeric Feature', border=1, ln=0, align='C')
    pdf.cell(w=42, h=10, txt=non_numeric_uniq_vals_df.columns[1], border=1, ln=1, align='C')

    # Table contents
    pdf.set_font('Arial', '', 12)
    for ii in range(0, min(5, len(non_numeric_uniq_vals_df))):
        pdf.cell(w=60, h=10,
                 txt=non_numeric_uniq_vals_df["Feature"].iloc[ii],
                 border=1, ln=0, align='L')
        pdf.cell(w=42, h=10,
                 txt=non_numeric_uniq_vals_df["Num Unique Values"].iloc[ii].astype(str),
                 border=1, ln=1, align='R')

    if len(numeric_uniq_vals_df) > 5:
        pdf.cell(w=0, h=10, txt="There are an additional {} non-numeric feature columns with more than 5 unique values.".format(len(non_numeric_uniq_vals_df) - 5), ln=1)

    return pdf


def section_on_feature_corr(pdf, numeric_df, numeric_collinear_df, non_numeric_df):

    pdf.add_page()

    pdf.set_font('Arial', 'B', 13)
    pdf.cell(w=0, h=10, txt="Feature Correlations", ln=1)

    # ---
    # Numeric feature correlations with Target variable

    pdf.ln(2)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(w=0, h=10, txt="Correlations of Numeric Features with Target Variable", ln=1)

    pdf.ln(2)

    # Table Header
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(w=60, h=10, txt='Numeric Feature', border=1, ln=0, align='C')
    pdf.cell(w=35, h=10, txt='Count non-Null', border=1, ln=0, align='C')
    pdf.cell(w=35, h=10, txt='Pearson Corr', border=1, ln=0, align='C')
    pdf.cell(w=35, h=10, txt='RF Corr', border=1, ln=1, align='C')

    # Table contents
    pdf.set_font('Arial', '', 12)
    for ii in range(0, min(10, len(numeric_df))):
        pdf.cell(w=60, h=10, txt=numeric_df.index[ii], border=1, ln=0, align='L')
        pdf.cell(w=35, h=10, txt=numeric_df["Count not-Null"].iloc[ii].astype(str), border=1, ln=0, align='R')
        pdf.cell(w=35, h=10, txt=numeric_df["Pearson"].iloc[ii].astype(str), border=1, ln=0, align='R')
        pdf.cell(w=35, h=10, txt=numeric_df["Random Forest"].iloc[ii].astype(str), border=1, ln=1, align='R')

    # ---
    # Correlations between numeric features

    pdf.ln(4)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(w=0, h=10, txt="Correlations between Numeric Features", ln=1)

    pdf.ln(2)

    # Table Header
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(w=52, h=10, txt='Numeric Feature 1', border=1, ln=0, align='C')
    pdf.cell(w=52, h=10, txt='Numeric Feature 2', border=1, ln=0, align='C')
    pdf.cell(w=32, h=10, txt='Count non-Null', border=1, ln=0, align='C')
    pdf.cell(w=28, h=10, txt='Pearson Corr', border=1, ln=0, align='C')
    pdf.cell(w=26, h=10, txt='RF Corr', border=1, ln=1, align='C')

    # Table contents
    pdf.set_font('Arial', '', 12)
    for ii in range(0, min(10, len(numeric_collinear_df))):
        pdf.cell(w=52, h=10, txt=numeric_collinear_df["Feature1"].iloc[ii], border=1, ln=0, align='L')
        pdf.cell(w=52, h=10, txt=numeric_collinear_df["Feature2"].iloc[ii], border=1, ln=0, align='L')
        pdf.cell(w=32, h=10, txt=numeric_collinear_df["Count not-Null"].iloc[ii].astype(str), border=1, ln=0, align='R')
        pdf.cell(w=28, h=10, txt=numeric_collinear_df["Pearson"].iloc[ii].astype(str), border=1, ln=0, align='R')
        pdf.cell(w=26, h=10, txt=numeric_collinear_df["Random Forest"].iloc[ii].astype(str), border=1, ln=1, align='R')

    # ---
    # Non-numeric feature correlations with Target variable

    pdf.add_page()

    pdf.set_font('Arial', 'B', 12)
    pdf.cell(w=0, h=10, txt="Correlations of Non-Numeric Features with Target Variable", ln=1)

    pdf.ln(2)

    # Table Header
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(w=60, h=10, txt='Non-Numeric Feature', border=1, ln=0, align='C')
    pdf.cell(w=32, h=10, txt='Count non-Null', border=1, ln=0, align='C')
    pdf.cell(w=28, h=10, txt='Num Unique', border=1, ln=0, align='C')
    pdf.cell(w=30, h=10, txt='RF Corr', border=1, ln=0, align='C')
    pdf.cell(w=30, h=10, txt='RF Corr (norm)', border=1, ln=1, align='C')

    # Table contents
    pdf.set_font('Arial', '', 12)
    for ii in range(0, min(10, len(non_numeric_df))):
        pdf.cell(w=60, h=10,
                 txt=non_numeric_df.index[ii],
                 border=1, ln=0, align='L')
        pdf.cell(w=32, h=10,
                 txt=non_numeric_df["Count not-Null"].iloc[ii].astype(str),
                 border=1, ln=0, align='R')
        pdf.cell(w=28, h=10,
                 txt=non_numeric_df["Num Unique"].iloc[ii].astype(str),
                 border=1, ln=0, align='R')
        pdf.cell(w=30, h=10,
                 txt=non_numeric_df["Random Forest"].iloc[ii].astype(str),
                 border=1, ln=0, align='R')
        pdf.cell(w=30, h=10,
                 txt=non_numeric_df["RF_norm"].iloc[ii].astype(str),
                 border=1, ln=1, align='R')

    return pdf


def section_of_plots(pdf, columns_list, target_col, numeric=True):

    pdf.add_page()
    pdf.set_font('Arial', 'B', 13)
    if numeric:
        pdf.cell(w=0, h=200, txt="Plots of Numeric Columns versus the Target Variable", ln=1, align='C')
    else:
        pdf.cell(w=0, h=200, txt="Plots of Non-Numeric Columns versus the Target Variable", ln=1, align='C')

    for jj, column in enumerate(columns_list):

        if (jj % 2) == 0:
            pdf.add_page()
        else:
            pdf.ln(4)

        # TODO: Double-check that file exists
        pdf.image('plots/{}_vs_{}.png'.format(column, target_col), x=10, y=None, w=180, h=0, type='PNG')

    return pdf


