# AI-Based Underwater Image Enhancement System
**SIH 2025 | Problem ID: 25243 | DRDO / Ministry of Defence**

---

## Project Structure
```
underwater-enhancement/
├── models/
│   ├── enhancement/
│   │   ├── generator.py        # U-Net GAN Generator
│   │   ├── discriminator.py    # PatchGAN Discriminator
│   │   └── losses.py           # Perceptual + SSIM + GAN loss
│   ├── superresolution/
│   │   └── esrgan.py           # 4x ESRGAN Super Resolution
│   └── knowledge_base/
│       └── submarine_db.py     # Submarine models + country database
├── pipeline/
│   ├── quality_gate.py         # Image quality assessment (UIQM)
│   ├── enhance.py              # Full enhancement pipeline
│   ├── detect.py               # Detection + honest reasoning
│   ├── comment_engine.py       # Structured comment generator
│   ├── metrics.py              # PSNR / SSIM / UIQM / UCIQE
│   └── report.py               # PDF operator report
├── training/
│   ├── synthetic_data.py       # Indian Ocean dataset generator
│   ├── train_enhancement.py    # Kaggle enhancement training
│   └── train_detection.py      # Kaggle YOLOv8 fine-tuning
├── dashboard/
│   ├── app.py                  # Streamlit dashboard
│   └── api.py                  # FastAPI backend
├── edge/
│   └── export_onnx.py          # ONNX export + benchmark
├── weights/                    # Place trained weights here
├── reports/                    # Generated PDF reports
└── requirements.txt
```

---

## Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Step 2: Train on Kaggle (Free GPU)

### 2a. Enhancement Model
1. Go to kaggle.com → New Notebook → Enable GPU T4
2. Upload your project files or clone repo
3. Add datasets:
   - Search: "EUVP underwater dataset" → Add to notebook
   - Search: "UIEB underwater image enhancement" → Add to notebook
4. Run in order:
```python
# Cell 1 — Generate synthetic Indian Ocean dataset
%run training/synthetic_data.py

# Cell 2 — Train enhancement model (~4-6 hours)
%run training/train_enhancement.py
```
5. Download: `/kaggle/working/models/best_generator.pt`

### 2b. Detection Model
1. New Kaggle Notebook → Enable GPU T4
2. Add dataset: "Brackish underwater dataset"
3. Run:
```python
%run training/train_detection.py
```
4. Download: `/kaggle/working/detection_training/underwater_yolo/weights/best.pt`

---

## Step 3: Place Weights
```
weights/
├── best_generator.pt       # From enhancement training
└── best_detection.pt       # From detection training (rename best.pt)
```

---

## Step 4: Run Dashboard
```bash
streamlit run dashboard/app.py
```

## Step 5: Run API (for external UI)
```bash
uvicorn dashboard.api:app --reload --port 8000
```
API docs available at: http://localhost:8000/docs

---

## Step 6: Export for Edge Deployment
```bash
python edge/export_onnx.py
```

---

## API Endpoints
| Endpoint | Method | Description |
|---|---|---|
| `/assess` | POST | Image quality assessment |
| `/enhance` | POST | Enhancement only |
| `/detect` | POST | Detection only |
| `/analyze` | POST | Full pipeline |
| `/report/{filename}` | GET | Download PDF |
| `/health` | GET | System status |

---

## Key Features
- Physics-based preprocessing (Beer-Lambert Indian Ocean model)
- U-Net GAN enhancement (haze, color, scatter removal)
- ESRGAN 4x super resolution for detail recovery
- YOLOv8n multi-class detection (11 classes)
- Submarine model + country inference (honest, not definitive)
- Honest confidence gating — never fakes high confidence
- Monte Carlo uncertainty quantification
- PDF operator report generation
- FastAPI backend for any external UI

## Important Note
Country identification is inference-based from visual features only.
NOT confirmed intelligence. Always requires human operator verification.
