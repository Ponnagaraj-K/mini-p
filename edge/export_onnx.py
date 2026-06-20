"""
Edge Deployment Export Script
Exports trained models to ONNX for CPU-optimized inference on edge devices.
Run after training on Kaggle.
"""
import torch
import numpy as np
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from models.enhancement.generator import get_generator

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False
    print("onnxruntime not installed. Run: pip install onnxruntime")


def export_generator_to_onnx(
    weights_path: str,
    output_path: str = "weights/generator.onnx",
    image_size: int = 256,
    device: str = "cpu"
) -> str:
    """Export enhancement generator to ONNX"""
    print(f"Loading generator from: {weights_path}")
    model = get_generator(weights_path, device)
    model.eval()

    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)

    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["degraded_image"],
        output_names=["enhanced_image"],
        dynamic_axes={
            "degraded_image": {0: "batch_size"},
            "enhanced_image": {0: "batch_size"}
        }
    )

    pt_size = Path(weights_path).stat().st_size / 1e6
    onnx_size = Path(output_path).stat().st_size / 1e6
    print(f"PyTorch model size: {pt_size:.2f} MB")
    print(f"ONNX model size:    {onnx_size:.2f} MB")
    print(f"Size reduction:     {pt_size/max(onnx_size, 0.001):.1f}x")
    return output_path


def benchmark_onnx(onnx_path: str, image_size: int = 256, n_runs: int = 50) -> dict:
    """Benchmark ONNX model inference speed"""
    if not ORT_AVAILABLE:
        return {"error": "onnxruntime not available"}

    print(f"\nBenchmarking ONNX model: {onnx_path}")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    dummy_input = np.random.randn(1, 3, image_size, image_size).astype(np.float32)

    # Warmup
    for _ in range(5):
        session.run(None, {"degraded_image": dummy_input})

    # Timed runs
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, {"degraded_image": dummy_input})
        times.append(time.perf_counter() - t0)

    avg_ms = np.mean(times) * 1000
    min_ms = np.min(times) * 1000
    fps = 1000 / avg_ms

    results = {
        "avg_inference_ms": round(avg_ms, 2),
        "min_inference_ms": round(min_ms, 2),
        "fps": round(fps, 2),
        "image_size": image_size,
        "n_runs": n_runs,
        "target_met": avg_ms < 100,
        "verdict": f"{'✅ MEETS' if avg_ms < 100 else '⚠️ BELOW'} <100ms target for edge deployment"
    }

    print(f"Average inference: {avg_ms:.2f}ms | FPS: {fps:.2f}")
    print(results["verdict"])
    return results


def export_yolo_to_onnx(weights_path: str, output_path: str = "weights/detection.onnx") -> str:
    """Export YOLOv8 detection model to ONNX"""
    try:
        from ultralytics import YOLO
        model = YOLO(weights_path)
        model.export(format="onnx", optimize=True, simplify=True)
        print(f"Detection model exported to ONNX")
        return output_path
    except Exception as e:
        print(f"YOLO ONNX export failed: {e}")
        return None


if __name__ == "__main__":
    Path("weights").mkdir(exist_ok=True)

    # Export enhancement model
    gen_weights = "weights/best_generator.pt"
    if Path(gen_weights).exists():
        onnx_path = export_generator_to_onnx(gen_weights, "weights/generator.onnx")
        benchmark_onnx(onnx_path)
    else:
        print(f"Enhancement weights not found at {gen_weights}")
        print("Train model on Kaggle first, then download weights.")

    # Export detection model
    det_weights = "weights/best_detection.pt"
    if Path(det_weights).exists():
        export_yolo_to_onnx(det_weights, "weights/detection.onnx")
    else:
        print(f"Detection weights not found at {det_weights}")
