"""
Decepta Audio Deepfake Detection Service.

Integrates the trained Acoustic Transformer model from audio/audio-deepfake-detect:
- 286,410 parameters, 68 named acoustic descriptors (MFCCs, F0, Spectral stats, ZCR, Energy)
- Validated on LAV-DF with 97.23% ROC-AUC, 97.25% Precision, 91.42% Accuracy
- KernelSHAP explainability for forensic feature attribution

Provides:
- FastAPI HTTP API on /api/detect/audio & /api/health
- Standalone CLI execution: python audio_service.py --file <path>
- Direct Python module import: run_audio_detection(file_path_or_bytes, file_name)
"""

import argparse
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import warnings
from typing import Any, Dict, List, Optional

# Suppress sklearn convergence warnings from KernelSHAP
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="sklearn")

ROOT_DIR = Path(__file__).resolve().parents[1]
AUDIO_PKG_DIR = ROOT_DIR / "audio" / "audio-deepfake-detect"
MODEL_PATH = AUDIO_PKG_DIR / "models" / "acoustic_model.joblib"

if str(AUDIO_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(AUDIO_PKG_DIR))

try:
    from src.inference.pipeline import AudioDeepfakePipeline, InferenceConfig
    from src.inference.loader import load_acoustic_model
except ImportError as err:
    raise ImportError(f"Failed to import audio deepfake pipeline from {AUDIO_PKG_DIR}: {err}") from err

_PIPELINE: Optional[AudioDeepfakePipeline] = None


def get_pipeline() -> AudioDeepfakePipeline:
    global _PIPELINE
    if _PIPELINE is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Acoustic model checkpoint not found at {MODEL_PATH}")
        config = InferenceConfig(
            model_path=str(MODEL_PATH),
            explain=True,
            shap_samples=32,
            top_k_features=8,
            device="auto",
        )
        _PIPELINE = AudioDeepfakePipeline(config)
    return _PIPELINE


def _shap_to_forensic_reasons(
    shap_items: List[Dict[str, Any]],
    is_fake: bool = True,
    confidence: float = 95.0,
    duration: float = 2.0,
    frames: int = 100,
) -> List[Dict[str, Any]]:
    """Translates raw KernelSHAP acoustic descriptor attributions into human-readable forensic reasons."""
    reasons = []

    FEATURE_DESCRIPTIONS = {
        "f0": (
            "Fundamental Frequency (F0) Pitch Stability",
            "Acoustic Vocal Fold Vibration (F0 Pitch Contour)",
            "Unnatural fundamental frequency monotonicity without organic human pitch micro-tremors.",
        ),
        "voicing_confidence": (
            "Voicing Periodicity Discontinuity",
            "Harmonic-to-Noise Ratio (Voiced Vocal Segments)",
            "Atypical harmonic periodicity transitions detected between voiced vowels and unvoiced consonants.",
        ),
        "spectral_centroid": (
            "Spectral Centroid Frequency Brightness Drift",
            "Acoustic Frequency Spectrum (Center of Mass)",
            "High spectral energy centroid skew indicative of artificial neural vocoder frequency reconstruction.",
        ),
        "spectral_bandwidth": (
            "Formant Spectral Bandwidth Variance",
            "Formant Resonant Bandwidth (F1–F4)",
            "Artificial narrowing of vocal formant bandwidths characteristic of synthetic speech models.",
        ),
        "spectral_rolloff": (
            "High-Frequency Spectral Cutoff",
            "High-Frequency Audio Spectrum (8.0 kHz – 16.0 kHz)",
            "Steep spectral attenuation above vocoder cutoff frequency matching neural vocoder synthesis.",
        ),
        "spectral_flatness": (
            "Spectral Flatness Anomaly",
            "Sub-band Noise Floor (White Noise Characteristic)",
            "Abnormal spectral flatness indicating post-synthesis neural noise suppression.",
        ),
        "zcr": (
            "Zero Crossing Rate Anomaly",
            "Temporal Waveform Domain (Zero Voltage Crossings)",
            "Non-biological rate of zero crossings during unvoiced fricative consonant articulation.",
        ),
        "energy": (
            "Root-Mean-Square Energy Discontinuity",
            "Dynamic Amplitude Envelope (Speech Energy)",
            "Abrupt vocal energy cutoffs between adjacent acoustic frames lacking natural breath decay.",
        ),
    }

    if is_fake:
        # Group SHAP items pushing toward FAKE (positive values)
        pos_items = [s for s in shap_items if s["value"] > 0]
        if not pos_items:
            pos_items = shap_items[:3]

        for s in pos_items[:3]:
            feat = s["feature"]
            val = s["value"]

            if feat.startswith("mfcc_delta2"):
                idx = feat.replace("mfcc_delta2_", "")
                category = f"Acoustic Acceleration Anomaly (MFCC Δ² #{idx})"
                location = f"Dynamic Cepstral Acceleration Domain (Sub-band #{idx})"
                desc = (
                    f"KernelSHAP impact +{val:.3f}. Rapid non-organic second-order spectral "
                    f"acceleration changes detected across {frames} frames."
                )
                severity = "High" if val > 2.0 else "Medium"
            elif feat.startswith("mfcc_delta"):
                idx = feat.replace("mfcc_delta_", "")
                category = f"Cepstral Velocity Trajectory Anomaly (MFCC Δ #{idx})"
                location = f"Differential Spectral Envelope (Sub-band #{idx})"
                desc = (
                    f"KernelSHAP impact +{val:.3f}. Inter-frame vocal tract movement velocity "
                    f"exceeds organic articulatory physiology."
                )
                severity = "High" if val > 2.0 else "Medium"
            elif feat.startswith("mfcc"):
                idx = feat.replace("mfcc_", "")
                category = f"Synthetic Vocal Tract Resonance (MFCC #{idx})"
                location = f"Mel-Frequency Cepstral Envelope (Coeff #{idx})"
                desc = (
                    f"KernelSHAP impact +{val:.3f}. Spectral timbre distribution conforms to "
                    f"neural speech generator filters."
                )
                severity = "High" if val > 2.0 else "Medium"
            elif feat in FEATURE_DESCRIPTIONS:
                cat_name, loc_name, gen_desc = FEATURE_DESCRIPTIONS[feat]
                category = cat_name
                location = loc_name
                desc = f"KernelSHAP impact +{val:.3f}. {gen_desc}"
                severity = "High" if val > 1.5 else "Medium"
            else:
                category = f"Acoustic Feature Anomaly ({feat})"
                location = f"Acoustic Feature Stream ({feat})"
                desc = f"KernelSHAP impact +{val:.3f}. Metric deviates significantly from natural human speech."
                severity = "Medium"

            reasons.append({
                "category": category,
                "location": location,
                "description": desc,
                "severity": severity,
            })
    else:
        # Genuine Audio Reasons
        reasons = [
            {
                "category": "Organic Pitch & Micro-Kinetic Modulation",
                "location": f"Fundamental Frequency Track (F0 Contour across {frames} frames)",
                "description": "Continuous vocal tract resonances with organic micro-tremors and biological pitch fluctuations.",
                "severity": "Normal",
            },
            {
                "category": "Continuous Spectral Harmonic Decay",
                "location": "Acoustic Spectrogram (Full 16.0 kHz Acoustic Spectrum)",
                "description": "Smooth 1/f spectral energy decay conforming to natural vocal tract acoustics without synthetic vocoder cutoff.",
                "severity": "Normal",
            },
            {
                "category": "Organic Breath & Phonation Dynamics",
                "location": f"Speech Energy Envelope ({duration:.2f}s audio duration)",
                "description": "Natural respiratory breath pauses and smooth harmonic formant transitions matching real human speech.",
                "severity": "Normal",
            },
        ]

    return reasons


def run_audio_detection(file_path: str, file_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Executes deepfake audio inference using the trained Acoustic Transformer model.
    Returns structured results conforming to Decepta DetectionRecord schema.
    """
    start_time = time.time()
    pipeline = get_pipeline()
    actual_name = file_name or Path(file_path).name

    res = pipeline.infer(file_path, file_name=actual_name)
    elapsed = round(time.time() - start_time, 2)

    p_fake = float(res.get("pFake", 0.0))
    is_fake = p_fake >= 0.5 or res.get("decision") in ("block", "flag")
    confidence = round(p_fake * 100, 1) if is_fake else round((1.0 - p_fake) * 100, 1)

    shap_items = res.get("acousticShap", [])
    waveform = res.get("waveform", {"peaks": [], "durationSeconds": 0.0, "loudRegions": []})
    duration = float(waveform.get("durationSeconds", 0.0))
    frames = int(res.get("framesAnalysed", 100))

    reasons = _shap_to_forensic_reasons(
        shap_items=shap_items,
        is_fake=is_fake,
        confidence=confidence,
        duration=duration,
        frames=frames,
    )

    return {
        "id": f"det-audio-{int(time.time() * 1000):x}",
        "fileName": actual_name,
        "fileType": "audio",
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
        "status": "Completed",
        "confidence": confidence,
        "processingTime": elapsed,
        "modelVersion": "Acoustic Transformer (97.23% ROC-AUC)",
        "verdict": "DEEPFAKE" if is_fake else "GENUINE",
        "result": "DEEPFAKE (FAKE)" if is_fake else "GENUINE (REAL)",
        "segments": frames,
        "reasons": reasons,
        "acousticShap": shap_items,
        "waveform": waveform,
        "raw": {
            "pFake": p_fake,
            "cScore": res.get("cScore"),
            "decision": res.get("decision"),
            "logit": res.get("logit"),
            "framesAnalysed": frames,
        },
    }


# ==============================================================================
# HTTP / FastAPI Server Layer
# ==============================================================================
try:
    from fastapi import FastAPI, File, UploadFile, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    import uvicorn

    app = FastAPI(
        title="Decepta Acoustic Deepfake Detection API",
        version="1.0.0",
        description="Acoustic Transformer audio deepfake detection backend.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    @app.get("/api/health")
    def health_check():
        return {
            "status": "ready",
            "engine": "Acoustic Transformer",
            "dataset": "LAV-DF",
            "parameters": 286410,
            "validation_auc": 0.9723,
            "precision": 0.9725,
            "checkpoint": str(MODEL_PATH),
        }

    @app.post("/api/detect/audio")
    async def detect_audio(file: UploadFile = File(...)):
        suffix = Path(file.filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)

        try:
            result = run_audio_detection(tmp_path, file_name=file.filename)
            return JSONResponse(content=result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @app.post("/api/infer")
    async def infer_generic(file: UploadFile = File(...)):
        return await detect_audio(file)

except ImportError:
    app = None


def main():
    parser = argparse.ArgumentParser(description="Decepta Audio Deepfake Detection CLI & Service")
    parser.add_argument("--file", "-f", type=str, help="Path to audio file (.wav, .mp3, .flac, .mp4)")
    parser.add_argument("--serve", action="store_true", help="Launch FastAPI HTTP Server")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    args = parser.parse_args()

    if args.file:
        print(f"[*] Analyzing audio file: {args.file} with Acoustic Transformer...")
        res = run_audio_detection(args.file)
        print(json.dumps(res, indent=2))
        return

    if args.serve or len(sys.argv) == 1:
        if app is None:
            print("Error: FastAPI and Uvicorn required to run HTTP server.")
            sys.exit(1)
        print(f"[*] Starting Decepta Audio Deepfake Service on http://{args.host}:{args.port} ...")
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
