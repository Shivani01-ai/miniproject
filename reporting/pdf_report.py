"""
Forensic PDF Audit Report Generator
Document Integrity Verification and Tampering Localization

Builds a comprehensive, publication-quality forensic PDF summary report
documenting cryptographic file fingerprint, pipeline parameters, visual evidence,
suspicious candidate region measurements, evidence crops, OCR transcript,
and scientific methodology limitations.
"""

from typing import Optional, Dict, Any, List
import io
import html
import hashlib
from datetime import datetime

from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
    HRFlowable
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to compute total page count and render running header & footer.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        # Running top header
        self.setFont("Helvetica", 7)
        self.setFillColor(colors.HexColor("#718096"))
        self.drawString(36, 762, "DOCUMENT INTEGRITY VERIFICATION & FORENSICS LAB — AUDIT RECORD")
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.5)
        self.line(36, 756, 612 - 36, 756)

        # Running bottom footer
        self.line(36, 32, 612 - 36, 32)
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 36, 22, page_str)
        self.drawString(
            36,
            22,
            "CONFIDENTIAL FORENSIC AUDIT REPORT — HEURISTIC & STATISTICAL INDICATORS ONLY"
        )
        self.restoreState()


def _format_size(size_bytes: int) -> str:
    """Format bytes into readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def _pil_to_reportlab_img(
    pil_img: Image.Image,
    max_w: float = 250.0,
    max_h: float = 200.0,
    keeper_list: Optional[List[io.BytesIO]] = None
) -> RLImage:
    """
    Converts a PIL Image into a ReportLab Image with aspect-ratio preserving bounds.
    Maintains an open BytesIO buffer in keeper_list to prevent memory collection.
    """
    buf = io.BytesIO()
    if pil_img.mode not in ("RGB", "L"):
        converted = pil_img.convert("RGB")
    else:
        converted = pil_img

    converted.save(buf, format="JPEG", quality=88)
    buf.seek(0)
    if keeper_list is not None:
        keeper_list.append(buf)

    w, h = converted.size
    aspect = float(h) / max(1.0, float(w))
    target_w = float(min(w, max_w))
    target_h = target_w * aspect
    if target_h > max_h:
        target_h = float(max_h)
        target_w = target_h / aspect

    return RLImage(buf, width=target_w, height=target_h)


def generate_pdf_report(
    file_name: str,
    file_bytes: bytes,
    image: Image.Image,
    ela_data: Optional[Dict[str, Any]] = None,
    ocr_text: Optional[str] = None,
    loc_data: Optional[Dict[str, Any]] = None,
    timestamp_str: Optional[str] = None
) -> bytes:
    """
    Generates a complete forensic audit PDF report in memory and returns its raw bytes.

    Args:
        file_name: Name of the uploaded file.
        file_bytes: Exact bytes of the uploaded file (for SHA-256 calculation).
        image: Original PIL Image of the document.
        ela_data: Optional dictionary containing ELA output {'image', 'quality', 'scale'}.
        ocr_text: Optional extracted text string from OCR.
        loc_data: Optional dictionary containing localization output
                  {'marked_image', 'regions', 'strategy', 'sensitivity', 'min_area', ...}.
        timestamp_str: Optional string representing inspection timestamp.

    Returns:
        bytes: Downloadable PDF binary data.
    """
    keeper_buffers: List[io.BytesIO] = []
    pdf_buffer = io.BytesIO()

    # 1. Compute Cryptographic Fingerprint
    sha256_hash = hashlib.sha256(file_bytes).hexdigest()
    file_size_formatted = _format_size(len(file_bytes))
    if not timestamp_str:
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    # Document Geometry: Page width 612 pt (Letter), Margins 36 pt (0.5 in), Printable width 540 pt
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=46,
        bottomMargin=46
    )

    # 2. Typography Styles
    base_styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=base_styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1A365D"),
        alignment=0,
        spaceAfter=3
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#4A5568"),
        alignment=0,
        spaceAfter=10
    )

    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=base_styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#1A365D"),
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#2D3748")
    )

    body_bold = ParagraphStyle(
        "ReportBodyBold",
        parent=body_style,
        fontName="Helvetica-Bold"
    )

    mono_style = ParagraphStyle(
        "ReportMono",
        parent=base_styles["Code"],
        fontName="Courier",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#2D3748")
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=base_styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.white,
        alignment=1
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#2D3748")
    )

    table_cell_center = ParagraphStyle(
        "TableCellCenter",
        parent=table_cell_style,
        alignment=1
    )

    disclaimer_style = ParagraphStyle(
        "DisclaimerText",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#742A2A")
    )

    story = []

    # -------------------------------------------------------------------------
    # Header Section
    # -------------------------------------------------------------------------
    story.append(Paragraph("Forensic Document Integrity Audit Report", title_style))
    story.append(
        Paragraph(
            "Automated Multi-Signal Document Verification & Tampering Localization System",
            subtitle_style
        )
    )
    story.append(Spacer(1, 4))

    # -------------------------------------------------------------------------
    # High-Visibility Forensic Disclaimer Banner
    # -------------------------------------------------------------------------
    disclaimer_html = (
        "<b>FORENSIC NOTICE &amp; LIMITATION OF EVIDENCE:</b><br/>"
        "This automated report summarizes statistical, compression error (ELA), and optical feature indicators. "
        "Any flagged candidate regions represent localized deviations from estimated document baseline properties; "
        "they <b>do NOT constitute legal proof of forgery or tampering</b>. High-contrast text, regional scripts, "
        "seals, and re-compression cycles naturally introduce baseline variations. Findings must be corroborated "
        "by qualified forensic examination and institutional records verification."
    )
    disclaimer_table = Table(
        [[Paragraph(disclaimer_html, disclaimer_style)]],
        colWidths=[540]
    )
    disclaimer_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF5F5")),
            ("BORDER", (0, 0), (-1, -1), 1, colors.HexColor("#FEB2B2")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ])
    )
    story.append(disclaimer_table)
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # Document Metadata & Cryptographic Hash Fingerprint
    # -------------------------------------------------------------------------
    story.append(Paragraph("1. Document Cryptographic Fingerprint & Metadata", section_heading))

    meta_rows = [
        [
            Paragraph("<b>File Name:</b>", body_style),
            Paragraph(html.escape(file_name), body_style),
            Paragraph("<b>Analysis Timestamp:</b>", body_style),
            Paragraph(html.escape(timestamp_str), body_style)
        ],
        [
            Paragraph("<b>File Size:</b>", body_style),
            Paragraph(f"{file_size_formatted} ({len(file_bytes):,} bytes)", body_style),
            Paragraph("<b>Resolution:</b>", body_style),
            Paragraph(f"{image.width} × {image.height} px", body_style)
        ],
        [
            Paragraph("<b>Color Mode:</b>", body_style),
            Paragraph(f"{image.mode} ({len(image.getbands())} bands)", body_style),
            Paragraph("<b>File Format:</b>", body_style),
            Paragraph(html.escape(image.format or "JPEG/PNG"), body_style)
        ],
        [
            Paragraph("<b>SHA-256 Hash:</b>", body_style),
            Paragraph(f"<code>{sha256_hash}</code>", mono_style),
            Paragraph("", body_style),
            Paragraph("", body_style)
        ]
    ]

    meta_table = Table(meta_rows, colWidths=[90, 180, 110, 160])
    meta_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("SPAN", (1, 3), (3, 3)),  # Span SHA-256 hash across columns
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # Inspection Pipeline Execution Status
    # -------------------------------------------------------------------------
    story.append(Paragraph("2. Forensic Inspection Modules Executed", section_heading))

    ela_executed = ela_data is not None and "image" in ela_data
    ocr_executed = ocr_text is not None
    loc_executed = loc_data is not None and "marked_image" in loc_data

    pipeline_rows = [
        [
            Paragraph("Pipeline Module", table_header_style),
            Paragraph("Status", table_header_style),
            Paragraph("Configured Parameters &amp; Details", table_header_style)
        ],
        [
            Paragraph("<b>Error Level Analysis (ELA)</b>", table_cell_style),
            Paragraph(
                "<font color='#276749'><b>EXECUTED</b></font>" if ela_executed else "<font color='#718096'>SKIPPED</font>",
                table_cell_center
            ),
            Paragraph(
                f"JPEG Recompression Quality: {ela_data.get('quality', 90)}, Scale Factor: {ela_data.get('scale', 15)}×"
                if ela_executed else "Module not executed prior to report generation.",
                table_cell_style
            )
        ],
        [
            Paragraph("<b>OCR Text Extraction</b>", table_cell_style),
            Paragraph(
                "<font color='#276749'><b>EXECUTED</b></font>" if ocr_executed else "<font color='#718096'>SKIPPED</font>",
                table_cell_center
            ),
            Paragraph(
                f"EasyOCR Engine (CPU mode). Detected {len(ocr_text.splitlines())} text lines. (Confidence scores not provided by current module)."
                if ocr_executed else "Module not executed prior to report generation.",
                table_cell_style
            )
        ],
        [
            Paragraph("<b>Candidate Localization</b>", table_cell_style),
            Paragraph(
                "<font color='#276749'><b>EXECUTED</b></font>" if loc_executed else "<font color='#718096'>SKIPPED</font>",
                table_cell_center
            ),
            Paragraph(
                f"Strategy: {loc_data.get('strategy', 'Contextual')}, Sensitivity Multiplier (k): {loc_data.get('sensitivity', 'N/A')}, Min Area: {loc_data.get('min_area', 'N/A')} px², Flagged Candidates: {len(loc_data.get('regions', []))}"
                if loc_executed else "Module not executed prior to report generation.",
                table_cell_style
            )
        ]
    ]

    pipeline_table = Table(pipeline_rows, colWidths=[150, 90, 300])
    pipeline_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )
    story.append(pipeline_table)
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # Visual Forensic Evidence Comparisons
    # -------------------------------------------------------------------------
    story.append(Paragraph("3. Visual Forensic Evidence Overview", section_heading))

    img_col_w = 175
    img_max_h = 135

    rl_orig = _pil_to_reportlab_img(image, max_w=img_col_w, max_h=img_max_h, keeper_list=keeper_buffers)

    if ela_executed and isinstance(ela_data.get("image"), Image.Image):
        rl_ela = _pil_to_reportlab_img(ela_data["image"], max_w=img_col_w, max_h=img_max_h, keeper_list=keeper_buffers)
        ela_caption = f"ELA Map (Q={ela_data.get('quality', 90)}, Scale={ela_data.get('scale', 15)}×)"
    else:
        rl_ela = Paragraph("<i>ELA map was not generated.</i>", table_cell_center)
        ela_caption = "ELA Residuals (Not Run)"

    if loc_executed and isinstance(loc_data.get("marked_image"), Image.Image):
        rl_loc = _pil_to_reportlab_img(loc_data["marked_image"], max_w=img_col_w, max_h=img_max_h, keeper_list=keeper_buffers)
        num_cands = len(loc_data.get("regions", []))
        loc_caption = f"Candidate Regions ({num_cands} flagged)"
    else:
        rl_loc = Paragraph("<i>Localization was not generated.</i>", table_cell_center)
        loc_caption = "Localization (Not Run)"

    evidence_view_table = Table(
        [
            [rl_orig, rl_ela, rl_loc],
            [
                Paragraph("<b>(A) Original Ingested Document</b>", table_cell_center),
                Paragraph(f"<b>(B) {ela_caption}</b>", table_cell_center),
                Paragraph(f"<b>(C) {loc_caption}</b>", table_cell_center)
            ]
        ],
        colWidths=[180, 180, 180]
    )
    evidence_view_table.setStyle(
        TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, 0), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F7FAFC")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(evidence_view_table)
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # Candidate Suspicious Regions Details & Evidence Crops
    # -------------------------------------------------------------------------
    story.append(Paragraph("4. Candidate Suspicious Regions &amp; Evidence Cuts", section_heading))

    if loc_executed:
        regions: List[Dict[str, Any]] = loc_data.get("regions", [])
        if len(regions) > 0:
            story.append(
                Paragraph(
                    f"A total of <b>{len(regions)}</b> candidate region(s) exhibited statistical compression error "
                    f"deviations requiring verification. Ranked in descending order of <b>Anomaly Impact Score</b>.",
                    body_style
                )
            )
            story.append(Spacer(1, 5))

            # Table of candidate regions
            region_rows = [
                [
                    Paragraph("Region ID", table_header_style),
                    Paragraph("Coordinates (X, Y)", table_header_style),
                    Paragraph("Size (W × H)", table_header_style),
                    Paragraph("Area (px²)", table_header_style),
                    Paragraph("Mean Error", table_header_style),
                    Paragraph("Peak Error", table_header_style),
                    Paragraph("Anomaly Impact", table_header_style),
                ]
            ]

            # List up to 10 candidates in the table
            for idx, reg in enumerate(regions[:10], start=1):
                region_name = reg.get("Region", f"Region #{idx}")
                x = reg.get("X", 0)
                y = reg.get("Y", 0)
                w = reg.get("Width", 0)
                h = reg.get("Height", 0)
                area = reg.get("Area", w * h)
                mean_err = reg.get("Mean Error", 0.0)
                peak_err = reg.get("Peak Error", 0.0)
                impact = reg.get("Anomaly Impact", 0.0)

                region_rows.append([
                    Paragraph(f"<b>{html.escape(str(region_name))}</b>", table_cell_center),
                    Paragraph(f"({x}, {y})", table_cell_center),
                    Paragraph(f"{w} × {h}", table_cell_center),
                    Paragraph(f"{area:,}", table_cell_center),
                    Paragraph(f"{mean_err:.2f}", table_cell_center),
                    Paragraph(f"{peak_err:.2f}", table_cell_center),
                    Paragraph(f"<b>{impact:.2f}</b>", table_cell_center),
                ])

            if len(regions) > 10:
                region_rows.append([
                    Paragraph(f"<i>... and {len(regions) - 10} additional candidate regions (omitted for report brevity)</i>", table_cell_style),
                    Paragraph("", table_cell_style),
                    Paragraph("", table_cell_style),
                    Paragraph("", table_cell_style),
                    Paragraph("", table_cell_style),
                    Paragraph("", table_cell_style),
                    Paragraph("", table_cell_style),
                ])

            reg_table = Table(region_rows, colWidths=[70, 80, 75, 75, 75, 75, 90])
            table_style_list = [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C5282")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
            if len(regions) > 10:
                table_style_list.append(("SPAN", (0, -1), (-1, -1)))
            reg_table.setStyle(TableStyle(table_style_list))
            story.append(reg_table)
            story.append(Spacer(1, 8))

            # Visual Evidence Crops of Top Candidates (Up to 4)
            top_crops_to_show = min(4, len(regions))
            story.append(
                Paragraph(
                    f"<b>Close-up Inspection of Top {top_crops_to_show} Flagged Evidence Crops:</b>",
                    body_bold
                )
            )
            story.append(Spacer(1, 4))

            crop_cells = []
            margin = 8
            for idx, reg in enumerate(regions[:top_crops_to_show], start=1):
                rx = reg.get("X", 0)
                ry = reg.get("Y", 0)
                rw = reg.get("Width", 0)
                rh = reg.get("Height", 0)

                x1 = max(0, rx - margin)
                y1 = max(0, ry - margin)
                x2 = min(image.width, rx + rw + margin)
                y2 = min(image.height, ry + rh + margin)

                try:
                    cropped_img = image.crop((x1, y1, x2, y2))
                    rl_crop = _pil_to_reportlab_img(
                        cropped_img,
                        max_w=120,
                        max_h=70,
                        keeper_list=keeper_buffers
                    )
                except Exception:
                    rl_crop = Paragraph("<i>Crop Error</i>", table_cell_center)

                cell_content = [
                    rl_crop,
                    Paragraph(f"<b>{reg.get('Region', f'#{idx}')}</b>", table_cell_center),
                    Paragraph(f"Impact: {reg.get('Anomaly Impact', 0.0):.1f} | Peak: {reg.get('Peak Error', 0.0):.1f}", table_cell_center)
                ]
                crop_cells.append(cell_content)

            # Pad up to multiple of 2 or 4 for clean grid display
            while len(crop_cells) < 4 and len(crop_cells) > 0 and len(crop_cells) % 2 != 0:
                crop_cells.append([Paragraph("", table_cell_center)])

            crop_rows = []
            if len(crop_cells) <= 2:
                crop_rows.append(crop_cells)
                c_widths = [270] * len(crop_cells)
            else:
                crop_rows.append([crop_cells[0], crop_cells[1]])
                crop_rows.append([crop_cells[2], crop_cells[3]])
                c_widths = [270, 270]

            crop_table = Table(crop_rows, colWidths=c_widths)
            crop_table.setStyle(
                TableStyle([
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FAFAFA")),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ])
            )
            story.append(crop_table)

        else:
            story.append(
                Paragraph(
                    "<b>No candidate suspicious regions were flagged</b> at current sensitivity parameters "
                    f"(Sensitivity Multiplier: {loc_data.get('sensitivity', 2.6)}, "
                    f"Min Area: {loc_data.get('min_area', 300)} px²). "
                    "<i>Note: The absence of detected localized anomalies indicates uniform compression behavior; "
                    "it does not certify document authenticity against uncompressed or high-generation digital alterations.</i>",
                    body_style
                )
            )
    else:
        story.append(
            Paragraph(
                "<i>Candidate region localization was not executed prior to report generation. "
                "Run the Anomaly Localization tab to inspect and delineate suspicious regions.</i>",
                body_style
            )
        )

    story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # Optical Character Recognition (OCR) Transcript
    # -------------------------------------------------------------------------
    story.append(Paragraph("5. Optical Character Recognition (OCR) Transcript", section_heading))

    if ocr_executed:
        if ocr_text and ocr_text.strip():
            ocr_lines = ocr_text.splitlines()
            story.append(
                Paragraph(
                    f"Extracted <b>{len(ocr_lines)}</b> lines of machine-readable text via EasyOCR. "
                    "<b>Notice:</b> Character and word confidence scores are not provided by the current EasyOCR module configuration.",
                    body_style
                )
            )
            story.append(Spacer(1, 4))

            # Render text box (showing up to 40 lines neatly)
            display_lines = ocr_lines[:40]
            escaped_text = "<br/>".join([html.escape(line) for line in display_lines])
            if len(ocr_lines) > 40:
                escaped_text += f"<br/><b>[... {len(ocr_lines) - 40} additional lines detected in full document transcript ...]</b>"

            ocr_box = Table(
                [[Paragraph(escaped_text, mono_style)]],
                colWidths=[540]
            )
            ocr_box.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
                    ("BORDER", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ])
            )
            story.append(ocr_box)
        else:
            story.append(
                Paragraph(
                    "<b>No readable printed text detected</b> by EasyOCR in the ingested document. "
                    "Verify image resolution, contrast, or orientation.",
                    body_style
                )
            )
    else:
        story.append(
            Paragraph(
                "<i>OCR text extraction was not executed prior to report generation. "
                "Run the OCR &amp; Text Consistency tab to perform optical character transcription.</i>",
                body_style
            )
        )

    story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # Forensic Limitations & Recommendations
    # -------------------------------------------------------------------------
    story.append(Paragraph("6. Forensic Analysis Limitations &amp; Verification Guidance", section_heading))
    limitations_html = (
        "<b>1. Compression History Sensitivity:</b> Error Level Analysis (ELA) operates by evaluating differences "
        "introduced by a standardized recompression cycle. If a document underwent multiple recompression saves "
        "or social media compression, error levels may become uniform or introduce deceptive artifacts.<br/>"
        "<b>2. Printed Text &amp; Complex Script Inconsistencies:</b> Genuine high-contrast elements, bold headlines, "
        "intricate Indic regional script loops (e.g. Kannada, Devanagari), and official stamps naturally exhibit "
        "elevated compression residuals. High anomaly scores in these regions do not indicate alteration.<br/>"
        "<b>3. Absence of Reference Template:</b> In the absence of an authoritative digital master copy or cryptographic "
        "public key signature, heuristic localization serves as a triage assistant for human verification, "
        "not an autonomous authenticity certifier.<br/>"
        "<b>4. Recommended Action:</b> Physical document review under UV/oblique light, institutional database verification, "
        "and multi-modal deep-learning analysis are strongly recommended for high-stakes credential verification."
    )
    story.append(Paragraph(limitations_html, body_style))

    # Build PDF with custom NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)

    pdf_bytes = pdf_buffer.getvalue()
    pdf_buffer.close()

    # Close any image buffers
    for b in keeper_buffers:
        try:
            b.close()
        except Exception:
            pass

    return pdf_bytes
