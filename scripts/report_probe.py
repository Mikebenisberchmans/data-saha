from app.reports.specification import generate_report
from app.reports.generator import generate_report_pdf

result = generate_report("Write a report on orders revenue by status")
pdf_path = generate_report_pdf(result, "sales_report.pdf")
print(f"PDF written to {pdf_path}")
