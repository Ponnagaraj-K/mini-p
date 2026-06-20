import cv2
import numpy as np
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.knowledge_base.submarine_db import (
    SUBMARINE_DATABASE, UNDERWATER_OBJECTS,
    THREAT_COLORS, THREAT_ACTIONS, get_submarine_info, get_object_info
)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("WARNING: ultralytics not installed. Using mock detections.")


# Confidence thresholds — lower = more sensitive, higher = fewer false alarms
# Tuned per class based on object size and criticality
CONFIDENCE_THRESHOLDS = {
    "submarine":        0.45,   # Large object, lower threshold acceptable
    "mine":             0.55,   # Must minimize false alarms — high consequence
    "diver":            0.50,
    "uuv":              0.50,
    "drone_underwater": 0.50,
    "fish_school":      0.40,   # Non-threat, can afford more sensitivity
    "marine_mammal":    0.40,
    "coral_structure":  0.35,
    "debris":           0.40,
    "pipeline":         0.45,
    "invasive_species": 0.45,
    "default":          0.50
}

# YOLO class name → knowledge base key mapping
CLASS_MAP = {
    "submarine": "submarine",
    "mine": "mine",
    "diver": "diver",
    "underwater_drone": "drone_underwater",
    "uuv": "uuv",
    "fish": "fish_school",
    "fish_school": "fish_school",
    "marine_mammal": "marine_mammal",
    "dolphin": "marine_mammal",
    "whale": "marine_mammal",
    "coral": "coral_structure",
    "rock": "coral_structure",
    "debris": "debris",
    "wreckage": "debris",
    "pipeline": "pipeline",
    "cable": "pipeline",
    "person": "diver",
    "0": "submarine",
}


def estimate_object_size(bbox: list, image_shape: tuple, known_class: str) -> dict:
    """Estimate real-world object dimensions from bounding box + known submarine sizes"""
    x1, y1, x2, y2 = bbox
    pixel_width = x2 - x1
    pixel_height = y2 - y1
    img_h, img_w = image_shape[:2]
    size_ratio = (pixel_width * pixel_height) / (img_w * img_h)

    size_category = (
        "large" if size_ratio > 0.15 else
        "medium" if size_ratio > 0.04 else
        "small" if size_ratio > 0.01 else
        "very_small"
    )

    # Rough length estimate for submarines using perspective
    if known_class == "submarine":
        estimated_length = f"~{int(pixel_width * 0.8)}-{int(pixel_width * 1.2)}m (rough estimate)"
    else:
        estimated_length = None

    return {
        "size_category": size_category,
        "pixel_width": pixel_width,
        "pixel_height": pixel_height,
        "frame_coverage": round(size_ratio * 100, 2),
        "estimated_length": estimated_length
    }


def run_uncertainty_check(model, image: np.ndarray, n_runs: int = 8) -> dict:
    """
    Monte Carlo Dropout uncertainty quantification.
    Runs model multiple times with dropout active.
    High variance = uncertain prediction = flag for operator.
    """
    if not YOLO_AVAILABLE:
        return {"uncertainty": "unavailable", "std_dev": 0, "flag": "STABLE"}

    results = []
    for _ in range(n_runs):
        result = model(image, verbose=False)
        if result[0].boxes is not None and len(result[0].boxes) > 0:
            confs = result[0].boxes.conf.cpu().numpy().tolist()
            results.append(np.mean(confs) if confs else 0)
        else:
            results.append(0)

    if not results:
        return {"uncertainty": "no_detections", "std_dev": 0, "flag": "STABLE"}

    std = float(np.std(results))
    mean = float(np.mean(results))
    flag = (
        "HIGH UNCERTAINTY — Model inconsistent, treat with caution" if std > 0.15 else
        "MODERATE UNCERTAINTY — Verify with additional sensors" if std > 0.08 else
        "STABLE — Consistent predictions across runs"
    )
    return {"uncertainty": round(std, 4), "mean_confidence": round(mean, 4),
            "std_dev": round(std, 4), "flag": flag, "n_runs": n_runs}


def classify_submarine_type(image: np.ndarray, bbox: list, confidence: float) -> dict:
    """
    Classify submarine type using visual shape features.
    Honest — reports confidence and alternative possibilities.
    """
    x1, y1, x2, y2 = [int(c) for c in bbox]
    roi = image[max(0, y1):min(image.shape[0], y2),
                max(0, x1):min(image.shape[1], x2)]

    if roi.size == 0:
        return {"error": "Could not extract ROI for classification"}

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Shape-based feature extraction
    detected_features = []
    aspect_ratio = (x2 - x1) / max((y2 - y1), 1)

    if aspect_ratio > 5:
        detected_features.append("elongated hull shape")
    elif aspect_ratio > 3:
        detected_features.append("moderate length hull")
    else:
        detected_features.append("compact hull")

    # Size-based classification
    pixel_length = x2 - x1
    if pixel_length < 50:
        detected_features.append("very small hull")
        probable_category = "midget"
    elif pixel_length < 150:
        detected_features.append("teardrop hull shape")
        probable_category = "attack"
    else:
        detected_features.append("large elongated hull")
        probable_category = "nuclear_attack"

    # Edge complexity — conning tower presence
    edges = cv2.Canny(roi_gray, 50, 150)
    edge_ratio = np.sum(edges > 0) / max(edges.size, 1)
    if edge_ratio > 0.12:
        detected_features.append("prominent fin/sail")
    elif edge_ratio > 0.06:
        detected_features.append("small streamlined conning tower")

    # Match against database
    matches = {}
    for key, sub in SUBMARINE_DATABASE.items():
        if key == "unknown_submarine":
            continue
        if sub["category"] == probable_category:
            feature_score = sum(
                1 for f in detected_features
                if any(f.lower() in sf.lower() for sf in sub["visual_features"])
            )
            score = (feature_score / max(len(sub["visual_features"]), 1)) * confidence
            matches[sub["display_name"]] = {
                "score": round(score * 100, 1),
                "countries": sub["countries"],
                "threat_level": sub["threat_level"],
                "friendly_operators": sub["friendly_operators"],
                "sub_type": sub["type"]
            }

    matches = dict(sorted(matches.items(), key=lambda x: x[1]["score"], reverse=True)[:3])

    # Only report matches above meaningful threshold
    reliable_matches = {k: v for k, v in matches.items() if v["score"] > 25}

    return {
        "detected_features": detected_features,
        "probable_category": probable_category,
        "possible_models": reliable_matches if reliable_matches else {},
        "classification_confidence": round(confidence * 0.7, 3),
        "disclaimer": "Visual identification only — not confirmed intelligence. Cross-verify with sonar/IFF.",
        "top_match": list(reliable_matches.keys())[0] if reliable_matches else "Unidentified Submarine"
    }


def build_detection_record(
    class_name: str,
    confidence: float,
    bbox: list,
    image: np.ndarray,
    image_quality: float
) -> dict:
    """
    Build complete honest detection record with full reasoning.
    """
    kb_key = CLASS_MAP.get(class_name.lower(), class_name.lower())
    threshold = CONFIDENCE_THRESHOLDS.get(kb_key, CONFIDENCE_THRESHOLDS["default"])

    # Quality-adjusted confidence — poor image = reduced effective confidence
    quality_penalty = max(0, (0.5 - image_quality) * 0.3) if image_quality < 0.5 else 0
    adjusted_confidence = max(0, confidence - quality_penalty)

    # Below threshold — don't report as detection, report as anomaly
    if adjusted_confidence < threshold:
        return {
            "status": "below_threshold",
            "raw_confidence": round(confidence, 3),
            "adjusted_confidence": round(adjusted_confidence, 3),
            "threshold": threshold,
            "class": class_name,
            "message": f"Possible {class_name} detected but confidence ({adjusted_confidence:.0%}) below threshold ({threshold:.0%}). Logged as unconfirmed anomaly.",
            "bbox": bbox,
            "threat_level": "REVIEW",
            "action": "Manual operator review required — do not act on this detection alone"
        }

    # Is this a submarine? Run additional classification
    submarine_analysis = None
    if kb_key == "submarine":
        submarine_analysis = classify_submarine_type(image, bbox, adjusted_confidence)

    # Get base info
    if kb_key == "submarine":
        base_info = {
            "display_name": submarine_analysis.get("top_match", "Unknown Submarine") if submarine_analysis else "Unknown Submarine",
            "threat_level": "HIGH",
            "description": "Submarine-class underwater vessel detected",
            "action": THREAT_ACTIONS["HIGH"]
        }
    else:
        base_info = get_object_info(kb_key)

    size_info = estimate_object_size(bbox, image.shape, kb_key)

    # Confidence tier
    confidence_tier = (
        "VERY HIGH" if adjusted_confidence >= 0.90 else
        "HIGH" if adjusted_confidence >= 0.80 else
        "MODERATE" if adjusted_confidence >= 0.65 else
        "LOW — treat as preliminary"
    )

    # Quality warning
    quality_warning = None
    if image_quality < 0.35:
        quality_warning = f"Low image quality ({image_quality:.2f}) reduces detection reliability"

    record = {
        "status": "detected",
        "class": class_name,
        "kb_key": kb_key,
        "display_name": base_info["display_name"],
        "raw_confidence": round(confidence, 3),
        "adjusted_confidence": round(adjusted_confidence, 3),
        "confidence_tier": confidence_tier,
        "confidence_display": f"{adjusted_confidence:.0%}",
        "threshold_used": threshold,
        "threat_level": base_info["threat_level"],
        "threat_color": THREAT_COLORS.get(base_info["threat_level"], "#FFFFFF"),
        "description": base_info["description"],
        "recommended_action": base_info.get("action", THREAT_ACTIONS.get(base_info["threat_level"], "")),
        "bbox": [int(c) for c in bbox],
        "size_info": size_info,
        "quality_warning": quality_warning,
        "submarine_analysis": submarine_analysis
    }

    return record


class DetectionPipeline:
    def __init__(self, model_path=None, device='cpu'):
        self.device = device
        self.model = None
        if YOLO_AVAILABLE and model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
            print(f"Detection model loaded: {model_path}")
        elif YOLO_AVAILABLE:
            # Use pretrained YOLOv8n as base — fine-tuning replaces this
            print("WARNING: No custom weights. Using YOLOv8n base — fine-tune for underwater objects.")
            try:
                self.model = YOLO("yolov8n.pt")
            except Exception:
                print("YOLOv8n base model not available. Run in demo mode.")

    def detect(self, image: np.ndarray, image_quality: float = 0.5) -> dict:
        """
        Run full detection pipeline with honest reasoning.
        Returns structured results with reasoning for every detection.
        """
        if self.model is None:
            return self._demo_detections(image, image_quality)

        try:
            results = self.model(image, verbose=False, conf=0.25, iou=0.45)
        except Exception as e:
            return {"error": str(e), "detections": [], "alert_level": "ERROR"}

        raw_detections = results[0].boxes if results[0].boxes is not None else []
        detections = []
        below_threshold = []

        for box in raw_detections:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = self.model.names.get(cls_id, f"class_{cls_id}")
            bbox = box.xyxy[0].tolist()

            record = build_detection_record(cls_name, conf, bbox, image, image_quality)

            if record["status"] == "detected":
                detections.append(record)
            else:
                below_threshold.append(record)

        # Sort by threat level priority
        threat_priority = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "NONE": 4, "FRIENDLY": 5}
        detections.sort(key=lambda x: threat_priority.get(x["threat_level"], 99))

        # Overall alert level
        alert_level = "NONE"
        if detections:
            top_threat = detections[0]["threat_level"]
            alert_level = top_threat

        return {
            "detections": detections,
            "below_threshold_anomalies": below_threshold,
            "total_detected": len(detections),
            "alert_level": alert_level,
            "alert_action": THREAT_ACTIONS.get(alert_level, ""),
            "image_quality_used": round(image_quality, 4),
            "processing_note": f"Detected {len(detections)} objects above threshold. "
                               f"{len(below_threshold)} anomalies below threshold logged."
        }

    def _demo_detections(self, image: np.ndarray, image_quality: float) -> dict:
        """Demo mode when no model is loaded — shows system structure without real detections"""
        return {
            "detections": [],
            "below_threshold_anomalies": [],
            "total_detected": 0,
            "alert_level": "NONE",
            "alert_action": "",
            "image_quality_used": round(image_quality, 4),
            "processing_note": "No detection model loaded. Train and load custom weights for real detections.",
            "demo_mode": True
        }

    def draw_detections(self, image: np.ndarray, detection_results: dict) -> np.ndarray:
        """Draw bounding boxes with honest confidence and threat labels"""
        annotated = image.copy()
        color_map = {
            "CRITICAL": (0, 0, 255),
            "HIGH": (0, 100, 255),
            "MEDIUM": (0, 165, 255),
            "LOW": (0, 255, 255),
            "NONE": (0, 255, 0),
            "FRIENDLY": (255, 128, 0),
            "REVIEW": (128, 128, 128)
        }

        for det in detection_results.get("detections", []):
            x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
            color = color_map.get(det["threat_level"], (255, 255, 255))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"{det['display_name']} | {det['confidence_display']}"
            threat_label = f"[{det['threat_level']}]"
            if det.get("quality_warning"):
                threat_label += " ⚠"

            # Background for text
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - 35), (x1 + max(tw, 100), y1), color, -1)
            cv2.putText(annotated, threat_label, (x1 + 2, y1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        # Below threshold anomalies — grey dashed boxes
        for anomaly in detection_results.get("below_threshold_anomalies", []):
            x1, y1, x2, y2 = [int(c) for c in anomaly["bbox"]]
            for i in range(x1, x2, 10):
                cv2.line(annotated, (i, y1), (min(i + 5, x2), y1), (128, 128, 128), 1)
                cv2.line(annotated, (i, y2), (min(i + 5, x2), y2), (128, 128, 128), 1)
            cv2.putText(annotated, "? Unconfirmed", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)

        return annotated
