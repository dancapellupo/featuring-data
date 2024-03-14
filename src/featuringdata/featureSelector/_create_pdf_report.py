
from fpdf import FPDF


def initialize_pdf_doc():
    pdf = FPDF()

    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(w=0, h=10, txt="Feature Selection with Recursive Model Training", ln=1)
    pdf.ln(2)

    return pdf


def add_plot_pdf(pdf, file_path, new_page=True):

    if new_page:
        pdf.add_page()
    else:
        pdf.ln(4)

    # TODO: Double-check that file exists
    pdf.image(file_path, x=10, y=None, w=180, h=0, type='PNG')

    return pdf


def save_pdf_doc(pdf, custom_filename='TrainingResults', timestamp=''):
    pdf.output('./{}_Model_Training_Report_{}.pdf'.format(custom_filename, timestamp), 'F')


