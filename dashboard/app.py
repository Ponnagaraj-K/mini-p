"""
Streamlit Dashboard — Underwater Enhancement & Threat Detection
Run: streamlit run dashboard/app.py
"""
import streamlit as st
import cv2
import numpy as np
import sys
import os
import tempfile
from pathlib import Path
from datetime import datetime
import time

sys.path.append(str(Path(__file__).parent.parent))
from pipeline.quality_gate import assess_quality
from pipeline.enhance import EnhancementPipeline
from pipeline.detect import DetectionPipeline
from pipeline.metrics import compute_all_metrics
from pipeline.comment_engine import build_full_report_comment
from pipeline.report import generate_pdf_report

# Page config
st.set_page_config(
    page_title="UnderwaterAI — Maritime Security",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS — dark military theme
st.markdown("""
<style>
    .stApp { background-color: #0a0e1a; color: #e0e0e0; }
    .main-title { 
        color: #00d4ff; font-size: 2rem; font-weight: bold;
        text-align: center; padding: 1rem 0;
        border-bottom: 2px solid #00d4ff; margin-bottom: 1.5rem;
    }
    .section-header { color: #00d4ff; font-size: 1.1rem; font-weight: bold; margin: 1rem 0 0.5rem; }
    .metric-box {
        background: #111827; border: 1px solid #1f2d3d;
        border-radius: 8px; padding: 1rem; text-align: center;
    }
    .threat-critical { color: #ff0000; font-weight: bold; font-size: 1.1rem; }
    .threat-high { color: #ff6600; font-weight: bold; }
    .threat-medium { color: #ffaa00; font-weight: bold; }
    .threat-low { color: #ffff00; }
    .threat-none { color: #00ff00; }
    .threat-friendly { color: #0080ff; font-weight: bold; }
    .detection-card {
        background: #111827; border-left: 4px solid #00d4ff;
        border-radius: 4px; padding: 1rem; margin: 0.5rem 0;
    }
    .warning-box {
        background: #2d1b1b; border: 1px solid #ff4444;
        border-radius: 6px; padding: 0.8rem; margin: 0.5rem 0;
        color: #ff9999;
    }
    .success-box {
        background: #1b2d1b; border: 1px solid #44ff44;
        border-radius: 6px; padding: 0.8rem; margin: 0.5rem 0;
        color: #99ff99;
    }
    .info-box {
        background: #1b1b2d; border: 1px solid #4444ff;
        border-radius: 6px; padding: 0.8rem; margin: 0.5rem 0;
        color: #9999ff;
    }
    div[data-testid="metric-container"] {
        background: #111827; border: 1px solid #1f2d3d;
        border-radius: 8px; padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading AI models...")
def load_models():
    device = "cpu"
    enh_weights = "weights/best_generator.pt"
    det_weights = "weights/best_detection.pt"
    enh = EnhancementPipeline(
        enhancement_weights=enh_weights if Path(enh_weights).exists() else None,
        device=device
    )
    det = DetectionPipeline(
        model_path=det_weights if Path(det_weights).exists() else None,
        device=device
    )
    return enh, det


def get_threat_style(threat_level: str) -> str:
    styles = {
        "CRITICAL": "threat-critical",
        "HIGH": "threat-high",
        "MEDIUM": "threat-medium",
        "LOW": "threat-low",
        "NONE": "threat-none",
        "FRIENDLY": "threat-friendly"
    }
    return styles.get(threat_level, "")


def display_quality_metrics(metrics: dict):
    uiqm_before = metrics.get('uiqm_before', 0) or 0
    uiqm_after  = metrics.get('uiqm_after', 0) or 0
    uiqm_delta  = metrics.get('uiqm_improvement', 0) or 0
    uciqe_before = metrics.get('uciqe_before', 0) or 0
    uciqe_after  = metrics.get('uciqe_after', 0) or 0
    uciqe_delta  = metrics.get('uciqe_improvement', 0) or 0

    cols = st.columns(6)
    cols[0].metric("UIQM Before",  f"{uiqm_before:.3f}",  help="Underwater Image Quality Measure")
    cols[1].metric("UIQM After",   f"{uiqm_after:.3f}",
                   delta=f"+{uiqm_delta:.3f}" if uiqm_delta > 0 else f"{uiqm_delta:.3f}")
    cols[2].metric("UCIQE Before", f"{uciqe_before:.3f}", help="Underwater Color Image Quality")
    cols[3].metric("UCIQE After",  f"{uciqe_after:.3f}",
                   delta=f"+{uciqe_delta:.3f}" if uciqe_delta > 0 else f"{uciqe_delta:.3f}")
    psnr_val = metrics.get('psnr')
    ssim_val = metrics.get('ssim')
    cols[4].metric("PSNR (dB)", f"{psnr_val:.2f}" if psnr_val else "N/A", help="Requires reference image")
    cols[5].metric("SSIM",      f"{ssim_val:.4f}" if ssim_val else "N/A", help="Requires reference image")

    verdict = metrics.get("enhancement_verdict", "")
    if "EXCELLENT" in verdict or "GOOD" in verdict:
        st.markdown(f'<div class="success-box">✅ {verdict}</div>', unsafe_allow_html=True)
    elif "MINIMAL" in verdict or "NO improvement" in verdict:
        st.markdown(f'<div class="warning-box">⚠️ {verdict}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="info-box">ℹ️ {verdict}</div>', unsafe_allow_html=True)


def display_detection_card(det: dict, idx: int):
    threat = det.get("threat_level", "NONE")
    style = get_threat_style(threat)
    conf = det.get("confidence_display", "0%")
    name = det.get("display_name", "Unknown")
    action = det.get("recommended_action", "")
    quality_warning = det.get("quality_warning", "")

    with st.expander(f"Detection #{idx+1}: {name} | {threat} | Confidence: {conf}", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Object:** {name}")
            st.markdown(f'**Threat Level:** <span class="{style}">{threat}</span>', unsafe_allow_html=True)
            st.markdown(f"**Confidence:** {conf} ({det.get('confidence_tier', '')})")
            st.markdown(f"**Description:** {det.get('description', '')}")
        with col2:
            size = det.get("size_info", {})
            if size:
                st.markdown(f"**Size:** {size.get('size_category', 'unknown')} ({size.get('frame_coverage', 0):.1f}% of frame)")
                if size.get("estimated_length"):
                    st.markdown(f"**Est. Length:** {size['estimated_length']}")

        if quality_warning:
            st.markdown(f'<div class="warning-box">⚠️ {quality_warning}</div>', unsafe_allow_html=True)

        # Submarine analysis
        sub = det.get("submarine_analysis")
        if sub:
            st.markdown("---")
            st.markdown("**🔍 SUBMARINE ANALYSIS**")
            features = sub.get("detected_features", [])
            if features:
                st.markdown(f"Visual features: *{', '.join(features)}*")
            st.markdown(f"Probable category: **{sub.get('probable_category', 'unknown')}**")
            st.markdown(f"Classification confidence: **{sub.get('classification_confidence', 0):.0%}**")

            possible_models = sub.get("possible_models", {})
            if possible_models:
                st.markdown("**Possible models (visual match — NOT confirmed):**")
                for model_name, model_info in possible_models.items():
                    countries = ", ".join(model_info["countries"][:4])
                    friendly = model_info.get("friendly_operators", [])
                    friendly_note = f" ⚠️ Friendly: {', '.join(friendly)}" if friendly else ""
                    st.markdown(
                        f"- **{model_name}** — {model_info['score']:.0f}% shape match | "
                        f"Countries: {countries}{friendly_note}"
                    )
            else:
                st.markdown("*Insufficient detail for model identification*")

            st.markdown(f'<div class="warning-box">⚠️ {sub.get("disclaimer", "")}</div>',
                       unsafe_allow_html=True)

        st.markdown(f'<div class="info-box">📋 **RECOMMENDED ACTION:** {action}</div>',
                   unsafe_allow_html=True)


def run_full_analysis(image: np.ndarray, apply_sr: bool, location: str, operator_id: str,
                      enh_pipeline, det_pipeline):
    results = {}

    with st.spinner("Step 1/5: Assessing image quality..."):
        quality = assess_quality(image)
        results["quality"] = quality

    if not quality["can_process"]:
        st.markdown(f'<div class="warning-box">⛔ {quality["message"]}<br>{quality["recommendation"]}</div>',
                   unsafe_allow_html=True)
        return None

    with st.spinner("Step 2/5: Enhancing image..."):
        t0 = time.time()
        enh_result = enh_pipeline.enhance(image, apply_sr=apply_sr)
        enh_time = time.time() - t0
        results["enhanced"] = enh_result["enhanced"]
        results["sr_enhanced"] = enh_result.get("sr_enhanced")
        results["enhancement_report"] = enh_result["report"]
        results["enhancement_time"] = enh_time

    with st.spinner("Step 3/5: Running object detection..."):
        t0 = time.time()
        quality_after = assess_quality(results["enhanced"])
        detection_results = det_pipeline.detect(
            results["enhanced"], image_quality=quality_after["uiqm_score"]
        )
        annotated = det_pipeline.draw_detections(results["enhanced"], detection_results)
        det_time = time.time() - t0
        results["detection_results"] = detection_results
        results["annotated"] = annotated
        results["quality_after"] = quality_after
        results["detection_time"] = det_time

    with st.spinner("Step 4/5: Computing metrics..."):
        metrics = compute_all_metrics(image, results["enhanced"])
        results["metrics"] = metrics

    with st.spinner("Step 5/5: Generating report..."):
        full_comment = build_full_report_comment(
            quality, results["enhancement_report"], detection_results, metrics
        )
        results["full_comment"] = full_comment

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = f"reports/report_{timestamp}.pdf"
        try:
            Path("reports").mkdir(exist_ok=True)
            generate_pdf_report(
                image, results["enhanced"], annotated, quality,
                results["enhancement_report"], detection_results, full_comment,
                metrics, pdf_path, location, operator_id
            )
            results["pdf_path"] = pdf_path
        except Exception as e:
            results["pdf_path"] = None
            results["pdf_error"] = str(e)

    return results


def main():
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        apply_sr = st.toggle("Super Resolution (4x)", value=False,
                              help="Recovers fine details — slower on CPU")
        location = st.text_input("Location / GPS", value="Indian Ocean")
        operator_id = st.text_input("Operator ID", value="OPR-001")
        st.markdown("---")
        st.markdown("### 📊 System Status")
        enh_status = "✅ Loaded" if Path("weights/best_generator.pt").exists() else "⚠️ Demo Mode"
        det_status = "✅ Loaded" if Path("weights/best_detection.pt").exists() else "⚠️ Demo Mode"
        st.markdown(f"Enhancement: {enh_status}")
        st.markdown(f"Detection: {det_status}")
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown(
            "AI-Based Underwater Image Enhancement System\n\n"
            "SIH 25243 | DRDO Maritime Security\n\n"
            "**Detects:** Submarines, Mines, Divers, UUVs, Marine life"
        )

    # Main header
    st.markdown('<div class="main-title">🌊 UNDERWATER AI — MARITIME SECURITY SYSTEM</div>',
               unsafe_allow_html=True)

    # Load models
    enh_pipeline, det_pipeline = load_models()

    # Input tabs
    tab1, tab2, tab3 = st.tabs(["📸 Image Analysis", "🎥 Video Analysis", "📋 About System"])

    with tab1:
        st.markdown('<div class="section-header">Upload Underwater Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload underwater image (JPG, PNG)",
            type=["jpg", "jpeg", "png"],
            key="image_upload"
        )

        if uploaded:
            file_bytes = np.frombuffer(uploaded.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if image is None:
                st.error("Could not read image. Please upload a valid JPG or PNG.")
                return

            # Quick quality preview
            with st.expander("📊 Input Image Quality Assessment", expanded=False):
                quality_preview = assess_quality(image)
                q_cols = st.columns(3)
                q_cols[0].metric("UIQM Score", f"{quality_preview['uiqm_score']:.3f}")
                q_cols[1].metric("Quality Level", quality_preview["quality_level"].upper())
                q_cols[2].metric("Active Degradations", len(quality_preview["active_degradations"]))
                if quality_preview["active_degradations"]:
                    st.markdown(f"**Degradations detected:** {', '.join(quality_preview['active_degradations'])}")

            if st.button("🚀 Run Full Analysis", type="primary", use_container_width=True):
                results = run_full_analysis(
                    image, apply_sr, location, operator_id,
                    enh_pipeline, det_pipeline
                )

                if results is None:
                    return

                # Alert banner
                alert = results["full_comment"].get("alert_level", "NONE")
                overall_msg = results["full_comment"].get("overall_alert", "")
                if alert in ("CRITICAL", "HIGH"):
                    st.markdown(f'<div class="warning-box">{overall_msg}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="success-box">{overall_msg}</div>', unsafe_allow_html=True)

                # Image comparison
                st.markdown('<div class="section-header">Image Enhancement Results</div>',
                           unsafe_allow_html=True)
                img_cols = st.columns(3)
                orig_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                enh_rgb = cv2.cvtColor(results["enhanced"], cv2.COLOR_BGR2RGB)
                ann_rgb = cv2.cvtColor(results["annotated"], cv2.COLOR_BGR2RGB)

                img_cols[0].image(orig_rgb, caption="Original Image", use_container_width=True)
                img_cols[1].image(enh_rgb, caption="Enhanced Image", use_container_width=True)
                img_cols[2].image(ann_rgb, caption="Detection Overlay", use_container_width=True)

                if results.get("sr_enhanced") is not None:
                    st.image(cv2.cvtColor(results["sr_enhanced"], cv2.COLOR_BGR2RGB),
                            caption="Super Resolution (4x) Enhanced", use_container_width=True)

                # Metrics
                st.markdown('<div class="section-header">Quality Metrics</div>', unsafe_allow_html=True)
                display_quality_metrics(results["metrics"])

                # Performance
                perf_cols = st.columns(2)
                perf_cols[0].metric("Enhancement Time", f"{results['enhancement_time']:.2f}s")
                perf_cols[1].metric("Detection Time", f"{results['detection_time']:.2f}s")

                # Detections
                st.markdown('<div class="section-header">Detection Results</div>', unsafe_allow_html=True)
                det_results = results["detection_results"]
                detections = det_results.get("detections", [])

                # Threat summary
                summary = results["full_comment"].get("threat_summary", {})
                sum_cols = st.columns(6)
                threat_labels = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE", "FRIENDLY"]
                for col, label in zip(sum_cols, threat_labels):
                    count = summary.get(label, 0)
                    col.metric(label, count)

                if detections:
                    for i, det in enumerate(detections):
                        display_detection_card(det, i)
                else:
                    if det_results.get("demo_mode"):
                        st.markdown(
                            '<div class="info-box">ℹ️ Demo mode — No detection weights loaded. '
                            'Train model on Kaggle and place weights in /weights/ folder.</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            '<div class="success-box">✅ No confirmed threats detected above threshold.</div>',
                            unsafe_allow_html=True
                        )

                # Below threshold anomalies
                anomalies = det_results.get("below_threshold_anomalies", [])
                if anomalies:
                    with st.expander(f"⚪ Unconfirmed Anomalies ({len(anomalies)} logged)", expanded=False):
                        for a in anomalies:
                            st.markdown(f"- {a.get('message', '')}")

                # Enhancement report
                with st.expander("🔧 Enhancement Pipeline Report", expanded=False):
                    steps = results["enhancement_report"].get("steps", {})
                    for step, status in steps.items():
                        icon = "✅" if "SUCCESS" in status else "⚠️" if "PARTIAL" in status or "FALLBACK" in status else "❌"
                        st.markdown(f"{icon} **{step.replace('_', ' ').title()}:** {status}")
                    warnings = results["enhancement_report"].get("warnings", [])
                    if warnings:
                        for w in warnings:
                            st.warning(w)

                # PDF Download
                st.markdown('<div class="section-header">Export Report</div>', unsafe_allow_html=True)
                if results.get("pdf_path") and Path(results["pdf_path"]).exists():
                    with open(results["pdf_path"], "rb") as f:
                        st.download_button(
                            "📄 Download Operator Report (PDF)",
                            data=f.read(),
                            file_name=Path(results["pdf_path"]).name,
                            mime="application/pdf",
                            use_container_width=True
                        )
                else:
                    st.warning("PDF generation failed. Check reportlab installation.")

                # Operator note
                st.markdown(
                    f'<div class="warning-box">⚠️ {results["full_comment"].get("operator_note", "")}</div>',
                    unsafe_allow_html=True
                )

    with tab2:
        st.markdown('<div class="section-header">Video / Live Stream Analysis</div>',
                   unsafe_allow_html=True)
        video_file = st.file_uploader("Upload video file", type=["mp4", "avi", "mov"], key="video")

        if video_file:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(video_file.read())
            tfile.close()

            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            st.info(f"Video: {total_frames} frames | {fps:.1f} FPS")

            frame_skip = st.slider("Process every N frames", 1, 30, 5,
                                   help="Higher = faster but less coverage")

            if st.button("▶️ Analyze Video", type="primary"):
                frame_display = st.empty()
                progress = st.progress(0)
                detection_log = []
                frame_idx = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_idx % frame_skip == 0:
                        quality = assess_quality(frame)
                        if quality["can_process"]:
                            enh_result = enh_pipeline.enhance(frame)
                            enhanced = enh_result["enhanced"]
                            det_results = det_pipeline.detect(
                                enhanced, image_quality=quality["uiqm_score"]
                            )
                            annotated = det_pipeline.draw_detections(enhanced, det_results)

                            if det_results.get("detections"):
                                for det in det_results["detections"]:
                                    detection_log.append({
                                        "frame": frame_idx,
                                        "time": f"{frame_idx/fps:.1f}s",
                                        "object": det["display_name"],
                                        "confidence": det["confidence_display"],
                                        "threat": det["threat_level"]
                                    })

                            frame_display.image(
                                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                caption=f"Frame {frame_idx} | UIQM: {quality['uiqm_score']:.2f}",
                                use_container_width=True
                            )

                    progress.progress(min(frame_idx / max(total_frames, 1), 1.0))
                    frame_idx += 1

                cap.release()
                os.unlink(tfile.name)

                st.success(f"Video analysis complete. {len(detection_log)} detections logged.")
                if detection_log:
                    st.markdown("**Detection Log:**")
                    for log_entry in detection_log:
                        threat_cls = get_threat_style(log_entry["threat"])
                        st.markdown(
                            f"- Frame {log_entry['frame']} ({log_entry['time']}): "
                            f"**{log_entry['object']}** | {log_entry['confidence']} | "
                            f'<span class="{threat_cls}">{log_entry["threat"]}</span>',
                            unsafe_allow_html=True
                        )

    with tab3:
        st.markdown("## About This System")
        st.markdown("""
        **AI-Based Underwater Image Enhancement System for Maritime Security**
        
        SIH 2025 | Problem ID: 25243 | Organization: DRDO / Ministry of Defence
        
        ### Pipeline
        1. **Quality Gate** — Assesses image quality (UIQM) before processing
        2. **Physics Preprocessing** — Beer-Lambert water absorption correction
        3. **Neural Enhancement** — U-Net GAN (FUnIE-GAN style) haze/scatter removal
        4. **Super Resolution** — ESRGAN 4x detail recovery
        5. **Object Detection** — YOLOv8n multi-class underwater detection
        6. **Submarine Classifier** — Type/model identification with country inference
        7. **Honest Reasoning** — Confidence gates, uncertainty quantification, disclaimers
        8. **Operator Report** — PDF export with full reasoning chain
        
        ### Detectable Objects
        - 🔴 Submarines (with model/country inference)
        - 🔴 Mines
        - 🟠 Unauthorized Divers
        - 🟠 UUVs / Underwater Drones
        - 🟢 Marine life (fish, mammals, coral)
        - 🔵 Infrastructure (pipelines, cables)
        
        ### Important Disclaimer
        > This system provides AI-assisted analysis only. Country identification
        > is inference-based from visual features — NOT confirmed intelligence.
        > All HIGH/CRITICAL detections require human operator confirmation.
        > Do not take autonomous action based on this system alone.
        """)


if __name__ == "__main__":
    main()
