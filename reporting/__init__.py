"""
Reporting package for Document Integrity Verification.
Builds downloadable PDF summary reports containing document metadata,
anomaly scores, text extracts, and highlighted crops.
"""

from reporting.pdf_report import generate_pdf_report

__all__ = ["generate_pdf_report"]

