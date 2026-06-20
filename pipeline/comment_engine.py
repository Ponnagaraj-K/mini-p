"""
Comment Engine — Generates honest, structured, operator-grade comments
for every detection. Never fabricates confidence or definitive IDs.
"""
from models.knowledge_base.submarine_db import THREAT_ACTIONS, SUBMARINE_DATABASE


def build_quality_comment(quality_assessment: dict) -> str:
    level = quality_assessment["quality_level"]
    uiqm = quality_assessment["uiqm_score"]
    active = quality_assessment.get("active_degradations", [])
    degradation_str = ", ".join(active) if active else "none detected"

    comments = {
        "unrecoverable": f"⛔ IMAGE UNRECOVERABLE | UIQM: {uiqm:.3f}\n"
                         f"Active degradations: {degradation_str}\n"
                         f"Analysis halted — insufficient data for reliable detection.\n"
                         f"Action: {quality_assessment['recommendation']}",
        "very_poor":     f"🔴 IMAGE QUALITY: VERY POOR | UIQM: {uiqm:.3f}\n"
                         f"Active degradations: {degradation_str}\n"
                         f"Enhancement applied. Results have high uncertainty.\n"
                         f"Action: {quality_assessment['recommendation']}",
        "poor":          f"🟠 IMAGE QUALITY: POOR | UIQM: {uiqm:.3f}\n"
                         f"Active degradations: {degradation_str}\n"
                         f"Enhancement applied. Detections are preliminary.\n"
                         f"Action: {quality_assessment['recommendation']}",
        "moderate":      f"🟡 IMAGE QUALITY: MODERATE | UIQM: {uiqm:.3f}\n"
                         f"Active degradations: {degradation_str}\n"
                         f"Enhancement improved clarity. Reasonable detection reliability.",
        "good":          f"🟢 IMAGE QUALITY: GOOD | UIQM: {uiqm:.3f}\n"
                         f"Active degradations: {degradation_str}\n"
                         f"High confidence detection conditions.",
        "high":          f"✅ IMAGE QUALITY: HIGH | UIQM: {uiqm:.3f}\n"
                         f"Optimal conditions for detection."
    }
    return comments.get(level, f"Quality: {level} | UIQM: {uiqm:.3f}")


def build_enhancement_comment(enhancement_report: dict) -> str:
    steps = enhancement_report.get("steps", {})
    warnings = enhancement_report.get("warnings", [])
    lines = ["ENHANCEMENT PIPELINE REPORT:"]

    step_icons = {
        "SUCCESS": "✓",
        "PARTIAL": "⚠",
        "FAILED": "✗",
        "FALLBACK": "⚠",
        "SKIPPED": "—"
    }

    for step, status in steps.items():
        icon = next((v for k, v in step_icons.items() if status.startswith(k)), "?")
        lines.append(f"  {icon} {step.replace('_', ' ').title()}: {status}")

    if warnings:
        lines.append("WARNINGS:")
        for w in warnings:
            lines.append(f"  ⚠ {w}")

    return "\n".join(lines)


def build_detection_comment(detection: dict) -> str:
    """Build full honest comment for a single detection"""
    if detection["status"] == "below_threshold":
        return (
            f"⚪ UNCONFIRMED ANOMALY\n"
            f"Possible: {detection['class']}\n"
            f"Raw confidence: {detection['raw_confidence']:.0%} | "
            f"Threshold required: {detection['threshold']:.0%}\n"
            f"Status: Below confidence threshold — not logged as detection\n"
            f"Action: {detection['action']}"
        )

    threat_icons = {
        "CRITICAL": "🔴",
        "HIGH": "🟠",
        "MEDIUM": "🟡",
        "LOW": "🔵",
        "NONE": "🟢",
        "FRIENDLY": "🔷"
    }

    threat_icon = threat_icons.get(detection["threat_level"], "⚪")
    lines = [
        f"{threat_icon} DETECTION: {detection['display_name']}",
        f"Confidence: {detection['confidence_display']} ({detection['confidence_tier']})",
        f"Threat Level: {detection['threat_level']}",
        f"Description: {detection['description']}",
    ]

    # Size info
    size = detection.get("size_info", {})
    if size:
        lines.append(f"Object size: {size.get('size_category', 'unknown')} "
                     f"({size.get('frame_coverage', 0):.1f}% of frame)")
        if size.get("estimated_length"):
            lines.append(f"Estimated length: {size['estimated_length']}")

    # Quality warning
    if detection.get("quality_warning"):
        lines.append(f"⚠ WARNING: {detection['quality_warning']}")

    # Submarine-specific analysis
    sub = detection.get("submarine_analysis")
    if sub:
        lines.append("")
        lines.append("SUBMARINE ANALYSIS:")
        lines.append(f"  Visual features: {', '.join(sub.get('detected_features', []))}")
        lines.append(f"  Probable category: {sub.get('probable_category', 'unknown')}")
        lines.append(f"  Classification confidence: {sub.get('classification_confidence', 0):.0%}")

        possible_models = sub.get("possible_models", {})
        if possible_models:
            lines.append("  Possible models (visual match only):")
            for model_name, model_info in possible_models.items():
                lines.append(f"    • {model_name}: {model_info['score']:.0f}% shape match")
                lines.append(f"      Countries operating this class: {', '.join(model_info['countries'])}")
                friendly = model_info.get("friendly_operators", [])
                if friendly:
                    lines.append(f"      ⚠ Includes friendly operators: {', '.join(friendly)}")
        else:
            lines.append("  Possible models: Insufficient detail for model identification")

        lines.append(f"  ⚠ DISCLAIMER: {sub.get('disclaimer', '')}")

    lines.append("")
    lines.append(f"RECOMMENDED ACTION: {detection['recommended_action']}")

    return "\n".join(lines)


def build_full_report_comment(
    quality_assessment: dict,
    enhancement_report: dict,
    detection_results: dict,
    metrics: dict
) -> dict:
    """
    Build complete structured report for dashboard and PDF export.
    """
    detections = detection_results.get("detections", [])
    anomalies = detection_results.get("below_threshold_anomalies", [])
    alert_level = detection_results.get("alert_level", "NONE")

    # Summary
    threat_summary = {
        "CRITICAL": sum(1 for d in detections if d["threat_level"] == "CRITICAL"),
        "HIGH": sum(1 for d in detections if d["threat_level"] == "HIGH"),
        "MEDIUM": sum(1 for d in detections if d["threat_level"] == "MEDIUM"),
        "LOW": sum(1 for d in detections if d["threat_level"] == "LOW"),
        "NONE": sum(1 for d in detections if d["threat_level"] == "NONE"),
        "FRIENDLY": sum(1 for d in detections if d["threat_level"] == "FRIENDLY"),
    }

    # Overall alert message
    if alert_level in ("CRITICAL", "HIGH"):
        overall_message = (
            f"🔴 THREAT DETECTED — {alert_level} PRIORITY\n"
            f"{detection_results.get('alert_action', '')}\n"
            f"Human operator review REQUIRED before any action."
        )
    elif alert_level == "MEDIUM":
        overall_message = "🟡 SUSPICIOUS ACTIVITY — Flag for operator review"
    elif alert_level in ("LOW", "NONE"):
        overall_message = "🟢 NO SIGNIFICANT THREATS — Routine monitoring"
    else:
        overall_message = "⚪ ANALYSIS COMPLETE"

    return {
        "overall_alert": overall_message,
        "alert_level": alert_level,
        "quality_comment": build_quality_comment(quality_assessment),
        "enhancement_comment": build_enhancement_comment(enhancement_report),
        "detection_comments": [build_detection_comment(d) for d in detections],
        "anomaly_comments": [build_detection_comment(a) for a in anomalies],
        "threat_summary": threat_summary,
        "metrics": {
            "psnr": metrics.get("psnr", "N/A"),
            "ssim": metrics.get("ssim", "N/A"),
            "uiqm_before": metrics.get("uiqm_before", "N/A"),
            "uiqm_after": metrics.get("uiqm_after", "N/A"),
        },
        "total_detections": len(detections),
        "total_anomalies": len(anomalies),
        "operator_note": (
            "This system provides AI-assisted analysis only. "
            "All HIGH/CRITICAL detections require human operator confirmation. "
            "Country identification is inference-based from visual features only — "
            "NOT confirmed intelligence. Do not take autonomous action based on this system alone."
        )
    }
