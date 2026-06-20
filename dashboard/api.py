"""
FastAPI Backend — For external UI integration (React/Flutter/etc)
All pipeline functions exposed as clean REST endpoints.
Run: uvicorn dashboard.api:app --reload --port 8000
"""
import cv2
import numpy as np
import base64
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import sys

sys.path.append(str(Path(__file__).parent.parent))
from pipeline.quality_gate import assess_quality
from pipeline.enhance import EnhancementPipeline
from pipeline.detect import DetectionPipeline
from pipeline.metrics import compute_all_metrics
from pipeline.comment_engine import build_full_report_comment
from pipeline.report import generate_pdf_report

app = FastAPI(
    title="Underwater Enhancement AI API",
    description="AI-Based Underwater Image Enhancement & Threat Detection — SIH 25243",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load models once at startup
ENHANCEMENT_WEIGHTS = os.getenv("ENHANCEMENT_WEIGHTS", "weights/best_generator.pt")
DETECTION_WEIGHTS = os.getenv("DETECTION_WEIGHTS", "weights/best_detection.pt")
DEVICE = "cpu"

enhancement_pipeline = EnhancementPipeline(
    enhancement_weights=ENHANCEMENT_WEIGHTS if Path(ENHANCEMENT_WEIGHTS).exists() else None,
    device=DEVICE
)
detection_pipeline = DetectionPipeline(
    model_path=DETECTION_WEIGHTS if Path(DETECTION_WEIGHTS).exists() else None,
    device=DEVICE
)


def decode_image(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image format")
    return img


def encode_image(image: np.ndarray) -> str:
    _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buffer).decode("utf-8")


@app.get("/")
def root():
    return {
        "system": "AI Underwater Enhancement & Threat Detection",
        "version": "1.0.0",
        "project": "SIH 25243 — DRDO Maritime Security",
        "endpoints": ["/assess", "/enhance", "/detect", "/analyze", "/report", "/health"]
    }


@app.get("/health")
def health():
    return {
        "status": "online",
        "enhancement_model": "loaded" if Path(ENHANCEMENT_WEIGHTS).exists() else "demo_mode",
        "detection_model": "loaded" if Path(DETECTION_WEIGHTS).exists() else "demo_mode",
        "device": DEVICE,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/assess")
async def assess_image_quality(file: UploadFile = File(...)):
    """Assess image quality before enhancement"""
    img = decode_image(await file.read())
    quality = assess_quality(img)
    return JSONResponse(content=quality)


@app.post("/enhance")
async def enhance_image(
    file: UploadFile = File(...),
    apply_sr: bool = Query(False, description="Apply super resolution (slower)")
):
    """Enhance underwater image — returns enhanced image + processing report"""
    img = decode_image(await file.read())
    quality_before = assess_quality(img)

    if not quality_before["can_process"]:
        return JSONResponse(content={
            "success": False,
            "message": quality_before["message"],
            "recommendation": quality_before["recommendation"],
            "quality": quality_before
        })

    result = enhancement_pipeline.enhance(img, apply_sr=apply_sr)
    enhanced = result["enhanced"]
    quality_after = assess_quality(enhanced)
    metrics = compute_all_metrics(img, enhanced)

    return JSONResponse(content={
        "success": True,
        "original_image": encode_image(img),
        "enhanced_image": encode_image(enhanced),
        "sr_image": encode_image(result["sr_enhanced"]) if result.get("sr_enhanced") is not None else None,
        "quality_before": quality_before,
        "quality_after": quality_after,
        "metrics": metrics,
        "enhancement_report": result["report"]
    })


@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    """Run object detection on image — use enhanced image for best results"""
    img = decode_image(await file.read())
    quality = assess_quality(img)
    detection_results = detection_pipeline.detect(img, image_quality=quality["uiqm_score"])
    annotated = detection_pipeline.draw_detections(img, detection_results)

    return JSONResponse(content={
        "success": True,
        "annotated_image": encode_image(annotated),
        "detections": detection_results,
        "image_quality": quality["uiqm_score"]
    })


@app.post("/analyze")
async def full_analysis(
    file: UploadFile = File(...),
    apply_sr: bool = Query(False),
    location: str = Query("Unknown", description="GPS or named location"),
    operator_id: str = Query("AUTO")
):
    """
    Full pipeline: quality check → enhance → detect → comment → report
    Single endpoint for complete analysis.
    """
    file_bytes = await file.read()
    img = decode_image(file_bytes)

    # Step 1: Quality gate
    quality = assess_quality(img)
    if not quality["can_process"]:
        return JSONResponse(content={
            "success": False,
            "stage": "quality_gate",
            "message": quality["message"],
            "quality": quality
        })

    # Step 2: Enhance
    enh_result = enhancement_pipeline.enhance(img, apply_sr=apply_sr)
    enhanced = enh_result["enhanced"]

    # Step 3: Detect on enhanced image
    quality_after = assess_quality(enhanced)
    detection_results = detection_pipeline.detect(
        enhanced, image_quality=quality_after["uiqm_score"]
    )
    annotated = detection_pipeline.draw_detections(enhanced, detection_results)

    # Step 4: Metrics
    metrics = compute_all_metrics(img, enhanced)

    # Step 5: Full comment report
    full_comment = build_full_report_comment(
        quality, enh_result["report"], detection_results, metrics
    )

    # Step 6: Generate PDF
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = f"reports/report_{timestamp}.pdf"
    try:
        generate_pdf_report(
            img, enhanced, annotated, quality,
            enh_result["report"], detection_results, full_comment,
            metrics, pdf_path, location, operator_id
        )
        pdf_available = True
    except Exception as e:
        pdf_available = False
        pdf_path = None

    return JSONResponse(content={
        "success": True,
        "original_image": encode_image(img),
        "enhanced_image": encode_image(enhanced),
        "annotated_image": encode_image(annotated),
        "quality_before": quality,
        "quality_after": quality_after,
        "detection_results": detection_results,
        "metrics": metrics,
        "full_comment": full_comment,
        "pdf_available": pdf_available,
        "pdf_path": pdf_path,
        "timestamp": timestamp,
        "location": location,
        "operator_id": operator_id
    })


@app.get("/report/{filename}")
async def download_report(filename: str):
    """Download generated PDF report"""
    report_path = Path("reports") / filename
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(
        str(report_path),
        media_type="application/pdf",
        filename=filename
    )
