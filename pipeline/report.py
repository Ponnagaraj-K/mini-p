import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import tempfile
import os


THREAT_COLORS_PDF = {
    "CRITICAL": colors.red,
    "HIGH": colors.orange,
    "MEDIUM": colors.yellow,
    "LOW": colors.lightblue,
    "NONE": colors.lightgreen,
    "FRIENDLY": colors.blue
}


def save_temp_image(image: np.ndarray, suffix: str = ".jpg") -> str:
    """Save numpy image to temp file for PDF embedding"""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    cv2.imwrite(tmp.name, image)
    return tmp.name


def generate_pdf_report(
    original_image: np.ndarray,
    enhanced_image: np.ndarray,
    annotated_image: np.ndarray,
    quality_assessment: dict,
    enhancement_report: dict,
    detection_results: dict,
    full_comment: dict,
    metrics: dict,
    output_path: str = None,
    location: str = "Unknown",
    operator_id: str = "AUTO"
) -> str:
    """Generate complete PDF operator report"""

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(Path("reports") / f"report_{timestamp}.pdf")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle("Title", parent=styles["Title"],
                                  fontSize=16, textColor=colors.darkblue, alignment=TA_CENTER)
    header_style = ParagraphStyle("Header", parent=styles["Heading2"],
                                   fontSize=12, textColor=colors.darkblue)
    body_style = ParagraphStyle("Body", parent=styles["Normal"],
                                 fontSize=9, leading=14)
    warning_style = ParagraphStyle("Warning", parent=styles["Normal"],
                                    fontSize=9, textColor=colors.red)
    mono_style = ParagraphStyle("Mono", parent=styles["Code"],
                                 fontSize=8, leading=12)

    story = []
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    # Header
    story.append(Paragraph("UNDERWATER THREAT ANALYSIS REPORT", title_style))
    story.append(Paragraph("AI-Based Maritime Security System | DRDO Project 25243", 
                           ParagraphStyle("sub", parent=styles["Normal"], 
                                         alignment=TA_CENTER, fontSize=9, 
                                         textColor=colors.grey)))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.darkblue))
    story.append(Spacer(1, 0.1*inch))

    # Metadata table
    alert_level = full_comment.get("alert_level", "NONE")
    alert_color = THREAT_COLORS_PDF.get(alert_level, colors.grey)
    meta_data = [
        ["Timestamp:", timestamp_str, "Location:", location],
        ["Operator ID:", operator_id, "Alert Level:", alert_level],
        ["Total Detections:", str(full_comment.get("total_detections", 0)),
         "Anomalies Logged:", str(full_comment.get("total_anomalies", 0))]
    ]
    meta_table = Table(meta_data, colWidths=[1.2*inch, 2.5*inch, 1.2*inch, 1.5*inch])
    meta_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("BACKGROUND", (1, 1), (1, 1), alert_color),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.whitesmoke, colors.white]),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.15*inch))

    # Overall alert
    story.append(Paragraph("ANALYSIS SUMMARY", header_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
    overall_msg = full_comment.get("overall_alert", "").replace("\n", "<br/>")
    story.append(Paragraph(overall_msg, warning_style if alert_level in ("CRITICAL", "HIGH") else body_style))
    story.append(Spacer(1, 0.1*inch))

    # Images — original, enhanced, annotated
    story.append(Paragraph("IMAGE ANALYSIS", header_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
    img_paths = []
    try:
        img_w = 2.1 * inch
        img_h = 1.6 * inch
        orig_path = save_temp_image(original_image)
        enh_path = save_temp_image(enhanced_image)
        ann_path = save_temp_image(annotated_image)
        img_paths = [orig_path, enh_path, ann_path]

        img_row = [
            [RLImage(orig_path, width=img_w, height=img_h), Paragraph("Original Image", body_style)],
            [RLImage(enh_path, width=img_w, height=img_h), Paragraph("Enhanced Image", body_style)],
            [RLImage(ann_path, width=img_w, height=img_h), Paragraph("Detection Overlay", body_style)]
        ]
        img_table = Table([[img_row[0][0], img_row[1][0], img_row[2][0]],
                           [img_row[0][1], img_row[1][1], img_row[2][1]]],
                          colWidths=[2.2*inch, 2.2*inch, 2.2*inch])
        img_table.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
        story.append(img_table)
    except Exception as e:
        story.append(Paragraph(f"Image embedding failed: {e}", body_style))
    story.append(Spacer(1, 0.1*inch))

    # Quality + Metrics
    story.append(Paragraph("QUALITY METRICS", header_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
    m = metrics
    metrics_data = [
        ["Metric", "Before Enhancement", "After Enhancement", "Improvement"],
        ["UIQM (primary)", str(m.get("uiqm_before", "N/A")),
         str(m.get("uiqm_after", "N/A")),
         str(m.get("uiqm_improvement", "N/A"))],
        ["UCIQE", str(m.get("uciqe_before", "N/A")),
         str(m.get("uciqe_after", "N/A")),
         str(m.get("uciqe_improvement", "N/A"))],
        ["PSNR (dB)", "—", str(m.get("psnr", "N/A (no reference)")), "—"],
        ["SSIM", "—", str(m.get("ssim", "N/A (no reference)")), "—"],
    ]
    metrics_table = Table(metrics_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
    metrics_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
    ]))
    story.append(metrics_table)
    story.append(Paragraph(m.get("note", ""), 
                           ParagraphStyle("note", parent=styles["Normal"], 
                                         fontSize=7, textColor=colors.grey)))
    story.append(Spacer(1, 0.1*inch))

    # Detections
    story.append(Paragraph("DETECTION RESULTS", header_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))

    detections = detection_results.get("detections", [])
    if not detections:
        story.append(Paragraph("No confirmed detections above threshold.", body_style))
    else:
        for i, det in enumerate(detections):
            threat_color = THREAT_COLORS_PDF.get(det["threat_level"], colors.grey)
            det_data = [
                [f"Detection #{i+1}: {det['display_name']}"],
                ["Confidence:", det["confidence_display"],
                 "Threat:", det["threat_level"]],
                ["Description:", det["description"], "", ""],
                ["Action:", det["recommended_action"], "", ""],
            ]
            if det.get("quality_warning"):
                det_data.append(["⚠ Warning:", det["quality_warning"], "", ""])

            sub = det.get("submarine_analysis")
            if sub and sub.get("possible_models"):
                for model_name, model_info in sub["possible_models"].items():
                    det_data.append([
                        f"Possible model:",
                        f"{model_name} ({model_info['score']:.0f}% match)",
                        "Countries:",
                        ", ".join(model_info["countries"][:3])
                    ])

            det_table = Table(det_data, colWidths=[1.5*inch, 2*inch, 1*inch, 2*inch])
            det_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), threat_color),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white if det["threat_level"] not in ("LOW", "NONE") else colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("SPAN", (0, 0), (-1, 0)),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ]))
            story.append(det_table)
            story.append(Spacer(1, 0.08*inch))

    # Anomalies
    anomalies = detection_results.get("below_threshold_anomalies", [])
    if anomalies:
        story.append(Paragraph(f"UNCONFIRMED ANOMALIES ({len(anomalies)} logged)", header_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
        for anomaly in anomalies:
            story.append(Paragraph(
                f"• {anomaly.get('class', 'Unknown')} — "
                f"Confidence: {anomaly.get('raw_confidence', 0):.0%} "
                f"(threshold: {anomaly.get('threshold', 0):.0%}) — {anomaly.get('message', '')}",
                mono_style
            ))

    # Operator disclaimer
    story.append(Spacer(1, 0.15*inch))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.darkblue))
    story.append(Paragraph("IMPORTANT DISCLAIMER", header_style))
    story.append(Paragraph(full_comment.get("operator_note", ""), warning_style))
    story.append(Paragraph(
        f"Report generated: {timestamp_str} | System: AI Maritime Security v1.0 | "
        f"For authorized personnel only",
        ParagraphStyle("footer", parent=styles["Normal"], fontSize=7,
                      textColor=colors.grey, alignment=TA_CENTER)
    ))

    doc.build(story)

    # Cleanup temp files
    for p in img_paths:
        try:
            os.unlink(p)
        except Exception:
            pass

    return output_path
