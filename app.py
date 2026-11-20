"""
Document Integrity Verification and Tampering Localization
Streamlit Web Application - Phase 1 UI Dashboard
"""

import streamlit as st
from PIL import Image
import io
from forensics.ela import perform_ela
from forensics.localization import locate_suspicious_regions

@st.cache_resource(show_spinner=False)
def get_cached_ocr_reader():
    """
    Initializes the EasyOCR reader once and caches it in memory across Streamlit reruns.
    """
    from ocr.text_extraction import get_easyocr_reader
    return get_easyocr_reader(languages=["en"], gpu=False)

def format_file_size(size_bytes: int) -> str:
    """Helper function to format bytes into a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"

def main():
    # --------------------------------------------------------------------------
    # 1. Page Configuration
    # --------------------------------------------------------------------------
    st.set_page_config(
        page_title="Document Integrity Verifier",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # --------------------------------------------------------------------------
    # 2. Sidebar - Project Information & Guidelines
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.header("📋 Project Details")
        st.markdown("**Project Title:**")
        st.write("Document Integrity Verification and Tampering Localization")

        st.markdown("**Domain:**")
        st.write("AI / Computer Vision / Document Forensics")

        st.markdown("**Pipeline Workflow:**")
        st.markdown(
            """
            1. **Ingestion:** Upload document image
            2. **Forensics:** Error Level & Noise analysis
            3. **OCR:** Text extraction & layout check
            4. **Deep Learning:** Tamper feature inspection
            5. **Localization:** Visual anomaly overlay
            6. **Reporting:** Export audit summary (PDF)
            """
        )

        st.divider()
        st.caption("College AIML Mini-Project | Team of 4")

    # --------------------------------------------------------------------------
    # 3. Main Dashboard Header & Academic Disclaimer
    # --------------------------------------------------------------------------
    st.title("📄 Document Integrity Verification & Tampering Localization")
    st.markdown(
        """
        An automated forensic assistant to inspect scanned or photographed documents 
        (e.g., **invoices, certificates, and marksheets**) for signs of digital alterations, 
        splicing, and compression inconsistencies.
        """
    )

    # Academic & Forensic Disclaimer
    st.warning(
        """
        **⚠️ Forensic Disclaimer:**  
        Automated analysis detects statistical, compression, and visual **anomalies**. 
        These indicators suggest regions that may warrant manual verification; they **do not prove forgery** 
        or guarantee document authenticity without a verified original reference.
        """,
        icon="⚠️"
    )

    st.divider()

    # --------------------------------------------------------------------------
    # 4. Document Upload Section
    # --------------------------------------------------------------------------
    st.subheader("1. Document Ingestion")

    uploaded_file = st.file_uploader(
        label="Upload a document image to begin analysis (JPG, JPEG, or PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear scan or photograph of a marksheet, invoice, or certificate."
    )

    # --------------------------------------------------------------------------
    # 5. Document Inspection & Future Modules
    # --------------------------------------------------------------------------
    if uploaded_file is not None:
        # Load image via PIL
        try:
            image_bytes = uploaded_file.getvalue()
            image = Image.open(io.BytesIO(image_bytes))
            width, height = image.size
            file_size_str = format_file_size(len(image_bytes))

            # Display metadata cards
            st.success("Document uploaded successfully!", icon="✅")

            col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
            with col_meta1:
                st.metric(label="File Name", value=uploaded_file.name)
            with col_meta2:
                st.metric(label="Resolution", value=f"{width} × {height} px")
            with col_meta3:
                st.metric(label="File Size", value=file_size_str)
            with col_meta4:
                st.metric(label="Color Mode", value=f"{image.mode} ({len(image.getbands())} ch)")

            # Display uploaded document preview
            st.subheader("2. Document Preview")
            st.image(
                image,
                caption=f"Uploaded Document: {uploaded_file.name}",
                use_container_width=True
            )

            st.divider()

            # ------------------------------------------------------------------
            # 6. Pipeline Modules (Placeholders for upcoming stages)
            # ------------------------------------------------------------------
            st.subheader("3. Forensic Inspection Pipeline")

            tab_forensics, tab_ocr, tab_localization, tab_report = st.tabs([
                "🔬 Forensic Analysis",
                "🔤 OCR & Text Consistency",
                "🗺️ Anomaly Localization",
                "📑 Audit Report"
            ])

            with tab_forensics:
                st.markdown("### Error Level Analysis (ELA)")
                st.markdown(
                    """
                    Error Level Analysis identifies discrepancies in JPEG compression across the document.
                    When parts of a document are digitally edited, pasted, or resaved, their compression error levels
                    frequently differ from the unmodified regions.
                    """
                )

                col_ctrl1, col_ctrl2 = st.columns(2)
                with col_ctrl1:
                    ela_quality = st.slider(
                        "JPEG Resave Quality",
                        min_value=50,
                        max_value=100,
                        value=90,
                        step=1,
                        help="Quality level used for temporary recompression. Default is 90."
                    )
                with col_ctrl2:
                    ela_scale = st.slider(
                        "Visual Scale Factor",
                        min_value=5,
                        max_value=50,
                        value=15,
                        step=1,
                        help="Brightness amplification multiplier to make subtle compression differences visible."
                    )

                run_ela = st.button("Run ELA Analysis", type="primary")

                ela_key = f"ela_result_{uploaded_file.name}"
                if run_ela:
                    try:
                        with st.spinner("Performing Error Level Analysis..."):
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

                    st.markdown("#### ELA Visualization")
                    col_view1, col_view2 = st.columns(2)
                    with col_view1:
                        st.image(image, caption="Original Document", use_container_width=True)
                    with col_view2:
                        st.image(
                            ela_result_img,
                            caption=f"ELA Map (Quality: {result_data['quality']}, Scale: {result_data['scale']}×)",
                            use_container_width=True
                        )

                    # Cautious forensic explanation
                    st.info(
                        """
                        **Interpretation Guide:**
                        - **Uniformity:** In an unaltered document, similar textures and backgrounds usually produce a consistent, uniform error level.
                        - **Brighter Highlights:** Regions that appear noticeably brighter than surrounding text or paper indicate potential compression differences (e.g., pasted text, edited numbers, or modified stamps).
                        - **Caution:** High-contrast lines, dark printed borders, and colored logos naturally show higher error levels even in authentic documents. ELA indicates **compression anomalies**, not legal proof of forgery.
                        """
                    )

                    # Download button for ELA result
                    buf = io.BytesIO()
                    ela_result_img.save(buf, format="PNG")
                    st.download_button(
                        label="⬇️ Download ELA Result Image",
                        data=buf.getvalue(),
                        file_name=f"ela_{uploaded_file.name}.png",
                        mime="image/png"
                    )

                st.divider()
                st.markdown("**Coming Next in Forensic Analysis:**")
                st.markdown("- **Noise Inconsistency Mapping:** Spotting spliced or digitally smoothed patches.")
                st.markdown("- **Edge Sharpness Analysis:** Finding unnatural digital insertion boundaries.")

            with tab_ocr:
                st.markdown("### Optical Character Recognition (OCR)")
                st.markdown(
                    """
                    Extracts readable printed text from the uploaded document using EasyOCR.
                    This enables text search, auditing, and digital transcription.
                    """
                )

                run_ocr = st.button("Extract Text", type="primary")

                ocr_key = f"ocr_result_{uploaded_file.name}"
                if run_ocr:
                    try:
                        with st.spinner("Initializing EasyOCR (CPU mode) and extracting text..."):
                            reader = get_cached_ocr_reader()
                            from ocr.text_extraction import extract_text
                            extracted_text = extract_text(image, reader=reader)
                            st.session_state[ocr_key] = extracted_text
                    except ImportError:
                        st.error(
                            "⚠️ **EasyOCR is not installed** in your Python environment.\n\n"
                            "To enable OCR, please run: `pip install easyocr` in your PowerShell terminal."
                        )
                    except Exception as e:
                        st.error(f"Error performing OCR extraction: {e}")

                if ocr_key in st.session_state:
                    extracted_text = st.session_state[ocr_key]

                    if extracted_text and extracted_text.strip():
                        st.success(
                            f"Text extracted successfully ({len(extracted_text.splitlines())} lines detected).",
                            icon="✅"
                        )
                        edited_text = st.text_area(
                            label="Extracted Document Text (editable for verification or corrections):",
                            value=extracted_text,
                            height=250
                        )

                        # Download button for text file
                        st.download_button(
                            label="⬇️ Download Extracted Text (.txt)",
                            data=edited_text,
                            file_name=f"ocr_{uploaded_file.name}.txt",
                            mime="text/plain"
                        )
                    else:
                        st.warning(
                            "No readable text was detected in this document. "
                            "Please check image resolution, orientation, and contrast.",
                            icon="⚠️"
                        )

                    st.info(
                        """
                        **Scientific Notice:**  
                        OCR reads visual text characters for digital inspection and auditing. 
                        Text extraction by itself does not establish whether a document or its text is authentic.
                        """
                    )

                st.divider()
                st.markdown("**Coming Next in OCR Analysis:**")
                st.markdown("- **Baseline Alignment:** Flags misaligned or floating numbers/grades.")
                st.markdown("- **Font Dimension Consistency:** Checks for unexpected font size anomalies in key fields.")

            with tab_localization:
                st.markdown("### Candidate Region Localization (Forensic Review)")
                st.markdown(
                    """
                    Delineates candidate regions requiring manual review using local adaptive statistical thresholding 
                    on JPEG compression error signals (ELA), structural grid filtering, and proximity box merging.
                    """
                )

                # Strategy Selector (Preserves baseline for comparison)
                loc_strategy_label = st.radio(
                    "Detection Strategy:",
                    options=["Contextual Multi-Evidence (Enhanced)", "Baseline Area-Average (Legacy)"],
                    horizontal=True,
                    help="Contextual mode evaluates surrounding-region contrast and 1D text geometry to suppress ordinary print without missing altered patches. Baseline mode preserves the earlier area-average algorithm for direct comparison."
                )
                selected_method = "contextual" if "Contextual" in loc_strategy_label else "baseline"

                # Parameter controls
                col_param1, col_param2 = st.columns(2)
                with col_param1:
                    loc_sensitivity = st.slider(
                        "Sensitivity Multiplier (k-factor)",
                        min_value=1.5,
                        max_value=4.0,
                        value=2.6,
                        step=0.1,
                        help="Controls statistical anomaly threshold. Lower values (1.8-2.2) flag subtler variations; higher values (2.8-3.5) suppress genuine text noise."
                    )
                    loc_min_area = st.slider(
                        "Minimum Region Area (px²)",
                        min_value=150,
                        max_value=1200,
                        value=300,
                        step=50,
                        help="Filters out tiny noise specks and isolated single-character fragments below this area."
                    )
                with col_param2:
                    loc_merge_dist = st.slider(
                        "Proximity Merge Distance (px)",
                        min_value=10,
                        max_value=50,
                        value=25,
                        step=5,
                        help="Merges adjacent candidate fragments into cohesive word/field boxes rather than separate letter boxes."
                    )
                    loc_suppress_grid = st.checkbox(
                        "Suppress Linear Table Grid Borders",
                        value=True,
                        help="Automatically filters out thin horizontal and vertical table rules so they do not trigger false alarms."
                    )

                run_loc = st.button("Locate Suspicious Regions", type="primary")

                loc_key = f"loc_result_{uploaded_file.name}"
                if run_loc:
                    try:
                        with st.spinner("Analyzing compression error distributions and grouping candidate regions..."):
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
                    if num_regions > 0:
                        st.warning(
                            f"Identified **{num_regions}** candidate region(s) requiring review using **{active_strategy}**. "
                            f"*(Note: Candidate count is a heuristic index, not a tampering confidence score).* ",
                            icon="🔍"
                        )
                    else:
                        st.info(
                            f"No candidate regions were flagged by **{active_strategy}** at current sensitivity settings. "
                            "*(Important: This indicates no prominent localized compression discrepancy was detected; "
                            "it does not prove the document is authentic).* ",
                            icon="ℹ️"
                        )

                    # Side-by-side comparison
                    col_loc_orig, col_loc_marked = st.columns(2)
                    with col_loc_orig:
                        st.image(image, caption="Original Document", use_container_width=True)
                    with col_loc_marked:
                        st.image(
                            marked_doc,
                            caption=f"Annotated Document: Candidate Regions Requiring Review ({num_regions} flagged)",
                            use_container_width=True
                        )

                    # Measurements table
                    if num_regions > 0:
                        st.markdown("#### Candidate Regions Requiring Review (Ranked by Anomaly Impact)")
                        st.dataframe(regions_list, use_container_width=True)

                    # Download button for marked-up image
                    buf_marked = io.BytesIO()
                    marked_doc.save(buf_marked, format="PNG")
                    st.download_button(
                        label="⬇️ Download Marked-Up Document",
                        data=buf_marked.getvalue(),
                        file_name=f"localized_{uploaded_file.name}.png",
                        mime="image/png"
                    )

                    # Detailed scientific disclaimer and context
                    st.info(
                        """
                        **Forensic Review Guide & False-Positive Context:**  
                        - **Candidate Regions, Not Confirmed Forgery:** These marked boxes delineate areas where local compression error distributions differ noticeably from the document baseline.
                        - **Normal Document Structure Influences:** High-contrast printed text, regional scripts with intricate loops (such as Kannada or Devanagari), colored university seals, bold headings, and camera focus variations naturally exhibit elevated compression residuals and may appear as candidate boxes.
                        - **Parameter Sensitivity:** If an authentic document produces too many candidate regions, increase the *Sensitivity Multiplier* or *Minimum Region Area*. If inspecting subtle digital edits, lower the multiplier with the understanding that more genuine text will be flagged for review.
                        """
                    )

                st.divider()
                st.markdown("**Coming Next in Localization:**")
                st.markdown("- **Deep Learning Verification:** Pretrained neural network patch inspection for verified anomalies.")
                st.markdown("- **Multi-signal Fusion:** Combining OCR layout anomalies with noise and ELA maps.")

            with tab_report:
                st.info(
                    """
                    **[Coming Next] Exportable PDF Forensic Report:**
                    - **Cryptographic Hash:** SHA-256 fingerprint of the uploaded file.
                    - **Extracted Text Transcript:** Full text dump with confidence metrics.
                    - **Visual Evidence Crops:** Cropped views of identified suspicious regions.
                    - **Auditing Summary:** Ready-to-download summary for project reviews.
                    """
                )

        except Exception as e:
            st.error(f"Error reading image: {e}")

    else:
        # Guidance state when no document has been uploaded yet
        st.info("👆 Please upload a document image above to inspect its metadata and preview it.")

if __name__ == "__main__":
    main()
