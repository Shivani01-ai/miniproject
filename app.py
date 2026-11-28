"""
Document Integrity Verification and Tampering Localization
Forensic Analysis Dashboard - Clean Modern SaaS UI (Inspired by ZeroEssay Template)
"""

import streamlit as st
from PIL import Image
import io
import hashlib
from forensics.ela import perform_ela
from forensics.localization import locate_suspicious_regions
from reporting import generate_pdf_report


@st.cache_resource(show_spinner=False)
def get_cached_ocr_reader():
    """Initializes the EasyOCR reader once and caches it in memory."""
    from ocr.text_extraction import get_easyocr_reader
    return get_easyocr_reader(languages=["en"], gpu=False)


def format_file_size(size_bytes: int) -> str:
    """Formats bytes into a readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def apply_custom_theme():
    """Applies modern light-themed SaaS styling inspired by ZeroEssay without unwanted emojis."""
    st.markdown(
        """
        <style>
        /* Base page background */
        .stApp {
            background-color: #F8FAFC;
            color: #1E293B;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        /* Top header bar */
        header[data-testid="stHeader"] {
            background-color: #FFFFFF;
            border-bottom: 1px solid #F1F5F9;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #FFFFFF;
            border-right: 1px solid #E2E8F0;
            padding-top: 1.2rem;
        }

        /* Brand logo in sidebar */
        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 4px 8px 18px 8px;
            border-bottom: 1px solid #F1F5F9;
            margin-bottom: 18px;
        }
        .sidebar-brand-icon {
            width: 34px;
            height: 34px;
            border-radius: 8px;
            background: linear-gradient(135deg, #10B981 0%, #16A34A 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 800;
            font-size: 1.15rem;
            box-shadow: 0 2px 6px rgba(22, 163, 74, 0.25);
        }
        .sidebar-brand-text {
            line-height: 1.15;
        }
        .sidebar-brand-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: #0F172A;
            letter-spacing: -0.02em;
        }
        .sidebar-brand-sub {
            font-size: 0.74rem;
            color: #64748B;
            font-weight: 500;
        }

        /* Navigation items in sidebar */
        .nav-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 9px 12px;
            border-radius: 8px;
            font-size: 0.88rem;
            font-weight: 500;
            color: #475569;
            margin-bottom: 3px;
            text-decoration: none;
            transition: all 0.15s ease;
        }
        .nav-item:hover {
            background-color: #F8FAFC;
            color: #0F172A;
        }
        .nav-item.active {
            background-color: #EDFAF1;
            color: #16A34A;
            font-weight: 600;
            border-left: 3px solid #16A34A;
        }

        /* Sidebar promo/status card */
        .sidebar-status-card {
            background: #EDFAF1;
            border: 1px solid #D1FAE5;
            border-radius: 12px;
            padding: 14px;
            margin-top: 28px;
            margin-bottom: 16px;
        }
        .sidebar-status-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.74rem;
            color: #166534;
            font-weight: 700;
            letter-spacing: 0.04em;
            margin-bottom: 6px;
        }
        .sidebar-status-badge {
            background: #DCFCE7;
            color: #15803D;
            padding: 2px 8px;
            border-radius: 9999px;
            font-size: 0.68rem;
            font-weight: 700;
        }
        .sidebar-status-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #0F172A;
        }
        .sidebar-status-sub {
            font-size: 0.74rem;
            color: #166534;
            margin-top: 2px;
        }
        .sidebar-status-btn {
            display: block;
            text-align: center;
            background: linear-gradient(135deg, #10B981 0%, #16A34A 100%);
            color: white !important;
            font-size: 0.8rem;
            font-weight: 600;
            padding: 8px;
            border-radius: 8px;
            margin-top: 10px;
            box-shadow: 0 2px 6px rgba(22, 163, 74, 0.25);
            text-decoration: none;
        }

        /* Sidebar user profile */
        .sidebar-user {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 12px 0 6px 0;
            border-top: 1px solid #F1F5F9;
            margin-top: 8px;
        }
        .user-avatar {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            background: #0F172A;
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.8rem;
        }
        .user-info {
            font-size: 0.8rem;
            line-height: 1.25;
        }
        .user-name {
            font-weight: 600;
            color: #0F172A;
        }
        .user-email {
            color: #64748B;
            font-size: 0.72rem;
        }

        /* Mode toggle pill */
        .mode-toggle {
            display: flex;
            gap: 6px;
            background: #F1F5F9;
            padding: 4px 6px;
            border-radius: 20px;
            font-size: 0.74rem;
            color: #64748B;
            width: fit-content;
            margin-top: 10px;
        }
        .mode-toggle span.active {
            background: white;
            color: #0F172A;
            padding: 2px 8px;
            border-radius: 14px;
            font-weight: 600;
            box-shadow: 0 1px 2px rgba(0,0,0,0.06);
        }

        /* Main Page Title */
        .main-title {
            font-size: 1.75rem;
            font-weight: 700;
            color: #0F172A;
            margin-bottom: 14px;
            letter-spacing: -0.025em;
        }

        /* User Instructions Banner (Exact visual structure from template) */
        .instruction-banner {
            background-color: #EDFAF1;
            border: 1px solid #D1FAE5;
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 18px;
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
        }
        .instruction-content h4 {
            font-size: 0.92rem;
            font-weight: 700;
            color: #0F172A;
            margin: 0 0 6px 0;
            letter-spacing: -0.01em;
        }
        .instruction-content ul {
            margin: 0;
            padding-left: 18px;
            font-size: 0.82rem;
            color: #334155;
            line-height: 1.55;
        }
        .instruction-badges {
            display: flex;
            gap: 8px;
            align-items: center;
        }
        .instruction-badge {
            background: white;
            border: 1px solid #BBF7D0;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.72rem;
            font-weight: 600;
            color: #15803D;
            box-shadow: 0 1px 2px rgba(0,0,0,0.02);
            white-space: nowrap;
        }

        /* Setting Toolbar */
        .setting-toolbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 16px;
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px 12px 0 0;
            font-size: 0.85rem;
            margin-top: 10px;
        }
        .setting-title {
            font-weight: 600;
            color: #334155;
            font-size: 0.85rem;
        }
        .setting-actions {
            display: flex;
            gap: 12px;
            align-items: center;
        }
        .setting-pill {
            background: #F1F5F9;
            border: 1px solid #E2E8F0;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.78rem;
            font-weight: 600;
            color: #334155;
        }
        .setting-pill.green {
            background: #EDFAF1;
            border-color: #BBF7D0;
            color: #15803D;
        }

        /* Central Workspace Card */
        .workspace-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-top: none;
            border-radius: 0 0 12px 12px;
            padding: 20px 22px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.03);
            margin-bottom: 20px;
        }

        /* Modern Segmented Tab Bar */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background-color: #F1F5F9;
            padding: 4px;
            border-radius: 10px;
            border: 1px solid #E2E8F0;
            margin-bottom: 18px;
        }
        .stTabs [data-baseweb="tab"] {
            color: #64748B;
            font-size: 0.86rem;
            font-weight: 500;
            padding: 7px 18px;
            border-radius: 7px;
            background-color: transparent;
            border: none;
            transition: all 0.15s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: #0F172A;
            background-color: rgba(255, 255, 255, 0.6);
        }
        .stTabs [aria-selected="true"] {
            color: #16A34A !important;
            background-color: #FFFFFF !important;
            font-weight: 600 !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
            border: 1px solid #E2E8F0 !important;
        }

        /* Primary Green Action Button (Matches template Submit button) */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #10B981 0%, #16A34A 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            padding: 8px 24px !important;
            font-size: 0.92rem !important;
            box-shadow: 0 2px 6px rgba(22, 163, 74, 0.25) !important;
            transition: all 0.15s ease !important;
        }
        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #059669 0%, #15803D 100%) !important;
            box-shadow: 0 4px 10px rgba(22, 163, 74, 0.35) !important;
            transform: translateY(-1px);
        }
        .stButton > button[kind="secondary"] {
            background-color: #FFFFFF !important;
            color: #334155 !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
            padding: 6px 16px !important;
            font-size: 0.88rem !important;
        }
        .stButton > button[kind="secondary"]:hover {
            border-color: #16A34A !important;
            color: #16A34A !important;
        }

        /* Image Display Frame */
        div[data-testid="stImage"] img {
            border-radius: 8px;
            border: 1px solid #E2E8F0;
            background-color: #F8FAFC;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }

        /* Evidence mount header */
        .evidence-mount-header {
            font-size: 0.78rem;
            font-weight: 700;
            color: #16A34A;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }

        /* Bottom Detectors Bar (Matches template bottom) */
        .detectors-bar {
            text-align: center;
            padding: 16px 8px;
            color: #64748B;
            font-size: 0.82rem;
            margin-top: 10px;
            border-top: 1px solid #E2E8F0;
        }
        .detectors-label {
            font-weight: 600;
            margin-bottom: 8px;
            color: #475569;
            font-size: 0.82rem;
        }
        .detectors-list {
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 18px;
            font-size: 0.74rem;
            font-weight: 600;
            color: #64748B;
        }
        .detector-item {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .detector-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #CBD5E1;
        }
        .detector-dot.active {
            background: #22C55E;
        }

        /* Compact light metadata strip */
        .meta-strip-light {
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            align-items: center;
            padding: 10px 16px;
            background-color: #F0FDF4;
            border: 1px solid #DCFCE7;
            border-radius: 8px;
            font-size: 0.82rem;
            color: #166534;
            margin-bottom: 16px;
        }
        .meta-strip-light strong {
            color: #0F172A;
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        }
        .meta-strip-light .sha {
            color: #15803D;
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        }

        /* Empty state notification box */
        .empty-state-box {
            background-color: #EFF6FF;
            border: 1px solid #BFDBFE;
            border-radius: 8px;
            padding: 14px 18px;
            font-size: 0.86rem;
            color: #1E40AF;
            margin-bottom: 14px;
        }

        /* Custom text area styling */
        .stTextArea textarea {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 8px !important;
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace !important;
        }
        .stTextArea textarea:focus {
            border-color: #16A34A !important;
            box-shadow: 0 0 0 1px #16A34A !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def main():
    # --------------------------------------------------------------------------
    # 1. Page Configuration
    # --------------------------------------------------------------------------
    st.set_page_config(
        page_title="DocVerify - Document Integrity Verification",
        page_icon="D",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    apply_custom_theme()

    # --------------------------------------------------------------------------
    # 2. Sidebar - Inspired by ZeroEssay Template Layout (Clean, No Emojis)
    # --------------------------------------------------------------------------
    with st.sidebar:
        # Brand Logo Header
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-brand-icon">D</div>
                <div class="sidebar-brand-text">
                    <div class="sidebar-brand-title">DocVerify</div>
                    <div class="sidebar-brand-sub">Document Forensics</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Navigation List with active pill styling (No Emojis)
        st.markdown(
            """
            <div class="nav-item">Document Intake</div>
            <div class="nav-item">ELA Analysis</div>
            <div class="nav-item active">Tamper Localization</div>
            <div class="nav-item">OCR Extraction</div>
            <div class="nav-item">Audit Summary</div>
            <div class="nav-item">Settings</div>
            """,
            unsafe_allow_html=True
        )

        # Status / Promo Card (Matches template's green bottom card)
        st.markdown(
            """
            <div class="sidebar-status-card">
                <div class="sidebar-status-header">
                    <span>FORENSIC ENGINE</span>
                    <span class="sidebar-status-badge">ACTIVE</span>
                </div>
                <div class="sidebar-status-title">CPU Mode</div>
                <div class="sidebar-status-sub">EasyOCR • ELA Forensics • ReportLab</div>
                <a class="sidebar-status-btn" href="#top">Verified Pipeline</a>
            </div>
            """,
            unsafe_allow_html=True
        )

        # User profile tile (Matches template's bottom profile)
        st.markdown(
            """
            <div class="sidebar-user">
                <div class="user-avatar">T4</div>
                <div class="user-info">
                    <div class="user-name">AIML Mini-Project</div>
                    <div class="user-email">Document Integrity Team</div>
                </div>
            </div>
            <div class="mode-toggle">
                <span class="active">Light</span>
                <span>Dark</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------------------------
    # 3. Main Dashboard Header & User Instructions Banner (No Emojis)
    # --------------------------------------------------------------------------
    st.markdown('<div class="main-title">Document Integrity Verification</div>', unsafe_allow_html=True)

    # Green User Instructions Banner (Matches template top card)
    st.markdown(
        """
        <div class="instruction-banner">
            <div class="instruction-content">
                <h4>User Instructions</h4>
                <ul>
                    <li>Upload a clear scan or photograph (JPG, JPEG, or PNG) in Document Intake.</li>
                    <li>Automated indicators highlight statistical anomalies for manual review — they do not prove forgery.</li>
                    <li>Switch freely between ELA Analysis, Tamper Localization, OCR Extraction, and Audit Summary.</li>
                </ul>
            </div>
            <div class="instruction-badges">
                <div class="instruction-badge">SHA-256 Digest</div>
                <div class="instruction-badge">EasyOCR Engine</div>
                <div class="instruction-badge">OpenCV Forensics</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Setting Toolbar (Matches template's Writing Setting toolbar)
    st.markdown(
        """
        <div class="setting-toolbar">
            <div class="setting-title">Inspection Settings</div>
            <div class="setting-actions">
                <span style="color: #64748B; font-size: 0.8rem;">Profile:</span>
                <div class="setting-pill">Standard Certificate ▾</div>
                <span style="color: #64748B; font-size: 0.8rem;">Method:</span>
                <div class="setting-pill green">Contextual Multi-Evidence ▾</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------------------------
    # 4. Central Workspace Card & 5 Functional Tabs (Zero Emojis)
    # --------------------------------------------------------------------------
    with st.container():
        st.markdown('<div class="workspace-card">', unsafe_allow_html=True)

        # 5 Clean Tabs without cartoon emojis
        tab_intake, tab_forensics, tab_localization, tab_ocr, tab_report = st.tabs([
            "Document Intake",
            "ELA Analysis",
            "Tamper Localization",
            "OCR Extraction",
            "Audit Summary"
        ])

        # ----------------------------------------------------------------------
        # Tab 1: Document Intake (Upload & Specimen Registration)
        # ----------------------------------------------------------------------
        with tab_intake:
            uploaded_file = st.file_uploader(
                label="Upload document image for forensic inspection (JPG, JPEG, or PNG):",
                type=["jpg", "jpeg", "png"],
                help="Upload a clear scan or photograph of a marksheet, invoice, or certificate."
            )

            if uploaded_file is not None:
                image_bytes = uploaded_file.getvalue()
                image = Image.open(io.BytesIO(image_bytes))
                width, height = image.size
                file_size_str = format_file_size(len(image_bytes))
                file_sha256 = hashlib.sha256(image_bytes).hexdigest()

                # Clean light metadata strip
                st.markdown(
                    f"""
                    <div class="meta-strip-light">
                        <span>File: <strong>{uploaded_file.name}</strong></span>
                        <span>Dimensions: <strong>{width} × {height} px</strong></span>
                        <span>Size: <strong>{file_size_str}</strong></span>
                        <span>Mode: <strong>{image.mode}</strong></span>
                        <span>SHA-256: <strong class="sha">{file_sha256[:16]}...</strong></span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown('<div class="evidence-mount-header">Ingested Specimen Preview</div>', unsafe_allow_html=True)
                st.image(image, use_container_width=True)

        # Retrieve current image if uploaded
        is_loaded = uploaded_file is not None

        # ----------------------------------------------------------------------
        # Tab 2: ELA Analysis
        # ----------------------------------------------------------------------
        with tab_forensics:
            if not is_loaded:
                st.markdown(
                    """
                    <div class="empty-state-box">
                        No document specimen is currently loaded. Please upload a document in Document Intake to perform Error Level Analysis.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                ela_key = f"ela_result_{uploaded_file.name}"

                col_btn, col_exp = st.columns([1, 3])
                with col_btn:
                    run_ela = st.button("Run ELA Analysis", type="primary")

                with st.expander("Advanced Parameters (Quality and Scale)", expanded=False):
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        ela_quality = st.slider("JPEG Resave Quality", 50, 100, 90, 1)
                    with col_p2:
                        ela_scale = st.slider("Visual Scale Factor", 5, 50, 15, 1)

                if run_ela:
                    try:
                        with st.spinner("Analyzing compression error levels..."):
                            ela_img = perform_ela(image, quality=ela_quality, scale=ela_scale)
                            st.session_state[ela_key] = {
                                "image": ela_img,
                                "quality": ela_quality,
                                "scale": ela_scale
                            }
                    except Exception as e:
                        st.error(f"Error executing ELA analysis: {e}")

                if ela_key in st.session_state:
                    result_data = st.session_state[ela_key]
                    ela_result_img = result_data["image"]

                    col_v1, col_v2 = st.columns(2)
                    with col_v1:
                        st.markdown('<div class="evidence-mount-header">Specimen A: Ingested Document</div>', unsafe_allow_html=True)
                        st.image(image, use_container_width=True)
                    with col_v2:
                        st.markdown(
                            f'<div class="evidence-mount-header">Evidence B: ELA Residual Map (Q={result_data["quality"]}, {result_data["scale"]}×)</div>',
                            unsafe_allow_html=True
                        )
                        st.image(ela_result_img, use_container_width=True)

                    buf = io.BytesIO()
                    ela_result_img.save(buf, format="PNG")
                    st.download_button(
                        label="Download ELA Result Image",
                        data=buf.getvalue(),
                        file_name=f"ela_{uploaded_file.name}.png",
                        mime="image/png"
                    )

                    with st.expander("Forensic Interpretation Guide", expanded=False):
                        st.caption("• Uniformity: Unaltered regions typically exhibit consistent compression error levels.")
                        st.caption("• Highlights: Noticeably brighter regions indicate potential recompression differences (e.g., pasted text or modified numbers).")
                        st.caption("• Caution: High-contrast printed lines, seals, and complex scripts naturally produce elevated residuals.")
                else:
                    st.markdown('<div class="evidence-mount-header">Specimen A: Ingested Document</div>', unsafe_allow_html=True)
                    st.image(image, use_container_width=True)

        # ----------------------------------------------------------------------
        # Tab 3: Tamper Localization
        # ----------------------------------------------------------------------
        with tab_localization:
            if not is_loaded:
                st.markdown(
                    """
                    <div class="empty-state-box">
                        No document specimen is currently loaded. Please upload a document in Document Intake to run tampering localization.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                loc_key = f"loc_result_{uploaded_file.name}"

                col_lbtn, col_lexp = st.columns([1, 3])
                with col_lbtn:
                    run_loc = st.button("Locate Suspicious Regions", type="primary")

                with st.expander("Localization Parameters and Strategy", expanded=False):
                    loc_strategy_label = st.radio(
                        "Detection Strategy:",
                        options=["Contextual Multi-Evidence (Enhanced)", "Baseline Area-Average (Legacy)"],
                        horizontal=True
                    )
                    selected_method = "contextual" if "Contextual" in loc_strategy_label else "baseline"

                    col_lp1, col_lp2 = st.columns(2)
                    with col_lp1:
                        loc_sensitivity = st.slider("Sensitivity Multiplier (k-factor)", 1.5, 4.0, 2.6, 0.1)
                        loc_min_area = st.slider("Minimum Region Area (px²)", 150, 1200, 300, 50)
                    with col_lp2:
                        loc_merge_dist = st.slider("Proximity Merge Distance (px)", 10, 50, 25, 5)
                        loc_suppress_grid = st.checkbox("Suppress Linear Table Grid Borders", value=True)

                if run_loc:
                    try:
                        with st.spinner("Localizing compression anomalies..."):
                            marked_img, detected_regions = locate_suspicious_regions(
                                image,
                                quality=90,
                                sensitivity=loc_sensitivity,
                                min_area=loc_min_area,
                                merge_distance=loc_merge_dist,
                                suppress_grid_lines=loc_suppress_grid,
                                method=selected_method
                            )
                            st.session_state[loc_key] = {
                                "marked_image": marked_img,
                                "regions": detected_regions,
                                "strategy": loc_strategy_label,
                                "sensitivity": loc_sensitivity,
                                "min_area": loc_min_area,
                                "merge_dist": loc_merge_dist,
                                "suppress_grid": loc_suppress_grid
                            }
                    except Exception as e:
                        st.error(f"Error performing region localization: {e}")

                if loc_key in st.session_state:
                    res_loc = st.session_state[loc_key]
                    marked_doc = res_loc["marked_image"]
                    regions_list = res_loc["regions"]
                    num_regions = len(regions_list)
                    active_strategy = res_loc.get("strategy", "Contextual Multi-Evidence")

                    col_lv1, col_lv2 = st.columns(2)
                    with col_lv1:
                        st.markdown('<div class="evidence-mount-header">Specimen A: Ingested Document</div>', unsafe_allow_html=True)
                        st.image(image, use_container_width=True)
                    with col_lv2:
                        st.markdown(
                            f'<div class="evidence-mount-header">Evidence C: Candidate Anomaly Overlay ({num_regions} Flagged)</div>',
                            unsafe_allow_html=True
                        )
                        st.image(marked_doc, use_container_width=True)

                    st.caption(
                        f"Strategy: {active_strategy} • {num_regions} candidate region(s) flagged. "
                        "(Candidate count is a heuristic index, not a tampering confidence score)."
                    )

                    buf_marked = io.BytesIO()
                    marked_doc.save(buf_marked, format="PNG")
                    st.download_button(
                        label="Download Annotated Document",
                        data=buf_marked.getvalue(),
                        file_name=f"localized_{uploaded_file.name}.png",
                        mime="image/png"
                    )

                    if num_regions > 0:
                        st.markdown("**Candidate Regions Table (Ranked by Anomaly Impact)**")
                        st.dataframe(regions_list, use_container_width=True)

                    with st.expander("Forensic Review and Sensitivity Context", expanded=False):
                        st.caption("• Candidate regions delineate areas where local compression error distributions differ noticeably from baseline.")
                        st.caption("• High-contrast printed text, intricate Indic script loops, and university seals naturally exhibit elevated residuals.")
                        st.caption("• If authentic documents produce too many candidates, increase the Sensitivity Multiplier or Minimum Area in the settings above.")
                else:
                    st.markdown('<div class="evidence-mount-header">Specimen A: Ingested Document</div>', unsafe_allow_html=True)
                    st.image(image, use_container_width=True)

        # ----------------------------------------------------------------------
        # Tab 4: OCR Extraction
        # ----------------------------------------------------------------------
        with tab_ocr:
            if not is_loaded:
                st.markdown(
                    """
                    <div class="empty-state-box">
                        No document specimen is currently loaded. Please upload a document in Document Intake to extract text.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                ocr_key = f"ocr_result_{uploaded_file.name}"

                run_ocr = st.button("Extract Text", type="primary")

                if run_ocr:
                    try:
                        with st.spinner("Extracting text via EasyOCR..."):
                            reader = get_cached_ocr_reader()
                            from ocr.text_extraction import extract_text
                            extracted_text = extract_text(image, reader=reader)
                            st.session_state[ocr_key] = extracted_text
                    except ImportError:
                        st.error("EasyOCR is not installed. Run `pip install easyocr` to enable OCR.")
                    except Exception as e:
                        st.error(f"Error performing OCR extraction: {e}")

                if ocr_key in st.session_state:
                    extracted_text = st.session_state[ocr_key]

                    if extracted_text and extracted_text.strip():
                        st.caption(f"Detected {len(extracted_text.splitlines())} text lines.")
                        edited_text = st.text_area(
                            label="Extracted Document Text (editable for verification):",
                            value=extracted_text,
                            height=240
                        )
                        st.download_button(
                            label="Download Transcript (.txt)",
                            data=edited_text,
                            file_name=f"ocr_{uploaded_file.name}.txt",
                            mime="text/plain"
                        )
                    else:
                        st.warning("No readable text detected in this document.")

                    with st.expander("OCR Methodology Notice", expanded=False):
                        st.caption("OCR extracts visual characters for transcription and auditing. Text presence alone does not establish authenticity.")
                else:
                    st.caption("Click 'Extract Text' above to transcribe printed characters from this document.")

        # ----------------------------------------------------------------------
        # Tab 5: Audit Summary (PDF Report)
        # ----------------------------------------------------------------------
        with tab_report:
            if not is_loaded:
                st.markdown(
                    """
                    <div class="empty-state-box">
                        No document specimen is currently loaded. Please upload a document in Document Intake to generate an audit report.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                ela_key = f"ela_result_{uploaded_file.name}"
                ocr_key = f"ocr_result_{uploaded_file.name}"
                loc_key = f"loc_result_{uploaded_file.name}"
                report_key = f"pdf_report_{uploaded_file.name}"

                ela_ready = ela_key in st.session_state
                ocr_ready = ocr_key in st.session_state
                loc_ready = loc_key in st.session_state

                loc_count_str = f"Ready ({len(st.session_state[loc_key]['regions'])} regions)" if loc_ready else "Not Run"
                st.markdown(
                    f"""
                    <div class="meta-strip-light">
                        <span>SHA-256: <strong class="sha">Ready</strong></span>
                        <span>ELA Map: <strong>{'Ready' if ela_ready else 'Not Run'}</strong></span>
                        <span>OCR Transcript: <strong>{'Ready' if ocr_ready else 'Not Run'}</strong></span>
                        <span>Localization: <strong>{loc_count_str}</strong></span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                generate_btn = st.button("Generate Verification Report", type="primary")

                if generate_btn:
                    try:
                        with st.spinner("Compiling PDF audit report..."):
                            ela_res = st.session_state.get(ela_key)
                            ocr_res = st.session_state.get(ocr_key)
                            loc_res = st.session_state.get(loc_key)

                            pdf_output = generate_pdf_report(
                                file_name=uploaded_file.name,
                                file_bytes=image_bytes,
                                image=image,
                                ela_data=ela_res,
                                ocr_text=ocr_res,
                                loc_data=loc_res
                            )
                            st.session_state[report_key] = pdf_output
                            st.success("Audit report generated successfully.")
                    except Exception as e:
                        st.error(f"Error generating audit report: {e}")

                if report_key in st.session_state:
                    generated_pdf = st.session_state[report_key]
                    pdf_size_str = format_file_size(len(generated_pdf))
                    base_name = uploaded_file.name.rsplit(".", 1)[0]
                    pdf_filename = f"forensic_report_{base_name}.pdf"

                    st.download_button(
                        label=f"Download PDF Audit Report ({pdf_size_str})",
                        data=generated_pdf,
                        file_name=pdf_filename,
                        mime="application/pdf",
                        type="primary"
                    )

                    with st.expander("Report Summary and Fingerprint Verification", expanded=False):
                        st.markdown(f"**Document File:** `{uploaded_file.name}`")
                        st.markdown(f"**SHA-256 Hash:** `{file_sha256}`")
                        st.markdown(f"**PDF Report Size:** `{pdf_size_str}` ({len(generated_pdf):,} bytes)")
                        st.caption("Report embeds document metadata, ELA residual map, annotated localization map, top candidate crops, and OCR transcripts.")

                st.caption(
                    "Note: Automated audit reports compile heuristic and statistical anomaly signals. "
                    "Flagged regions do not constitute definitive proof of tampering."
                )

        st.markdown('</div>', unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 5. Bottom Detectors Bar (Matches ZeroEssay Template Bottom)
    # --------------------------------------------------------------------------
    st.markdown(
        """
        <div class="detectors-bar">
            <div class="detectors-label">Integrated Forensic Verification Modules</div>
            <div class="detectors-list">
                <span class="detector-item"><span class="detector-dot active"></span> SHA-256 DIGEST</span>
                <span class="detector-item"><span class="detector-dot active"></span> ERROR LEVEL ANALYSIS</span>
                <span class="detector-item"><span class="detector-dot active"></span> EASYOCR ENGINE</span>
                <span class="detector-item"><span class="detector-dot active"></span> CONTEXTUAL LOCALIZATION</span>
                <span class="detector-item"><span class="detector-dot active"></span> GRID SUPPRESSION</span>
                <span class="detector-item"><span class="detector-dot active"></span> AUDIT EXPORT</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
