# utils.py
from fpdf import FPDF
import base64

class PDFReport(FPDF):
    def header(self):
        # Logo and Title
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'INFRALYTICS | Risk Audit Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_download_link(val, filename):
    b64 = base64.b64encode(val)
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}">📥 Download PDF Report</a>'

def generate_pdf_report(input_data, predictions, recommendations):
    pdf = PDFReport()
    pdf.add_page()
    
    # --- 1. PROJECT SUMMARY ---
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "1. Project Configuration", 0, 1)
    pdf.set_font("Arial", "", 10)
    
    # Iterate through inputs
    for key, value in input_data.items():
        pdf.cell(50, 8, f"{key.replace('_', ' ').title()}:", 0, 0)
        pdf.cell(0, 8, f"{str(value)}", 0, 1)
    
    pdf.ln(5)
    
    # --- 2. AI PREDICTIONS ---
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "2. AI Risk Assessment", 0, 1)
    pdf.set_font("Arial", "", 10)
    
    pdf.cell(50, 8, "Predicted Delay:", 0, 0)
    pdf.set_text_color(220, 50, 50) # Red
    pdf.cell(0, 8, f"{int(predictions['delay'])} Days", 0, 1)
    
    pdf.set_text_color(0, 0, 0) # Reset
    pdf.cell(50, 8, "Cost Overrun:", 0, 0)
    pdf.cell(0, 8, f"Rs. {int(predictions['cost'])} Lakhs", 0, 1)
    
    pdf.cell(50, 8, "Risk Severity:", 0, 0)
    pdf.cell(0, 8, f"{int(predictions['score'])} / 100", 0, 1)
    
    pdf.ln(5)

    # --- 3. RECOMMENDATIONS ---
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "3. Mitigation Strategy", 0, 1)
    pdf.set_font("Arial", "", 10)
    
    for rec in recommendations:
        pdf.set_font("Arial", "B", 10)
        pdf.cell(0, 8, f"- {rec['type']}", 0, 1)
        pdf.set_font("Arial", "", 9)
        pdf.multi_cell(0, 5, rec['msg'])
        pdf.ln(2)
        
    return pdf.output(dest="S").encode("latin-1")