import io
import torch
from PIL import Image, ImageChops
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_ID = "zodumair/document-forgery-detector"

class DocumentForgeryDetector:
    """Document-specific forgery/tampering screening model."""
    _processor = None
    _model = None

    @classmethod
    def _load(cls):
        if cls._model is None:
            cls._processor = AutoImageProcessor.from_pretrained(MODEL_ID)
            cls._model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
            cls._model.eval()
        return cls._processor, cls._model

    @staticmethod
    def _ela(image: Image.Image, quality=90, scale=15.0):
        original = image.convert("RGB")
        buf = io.BytesIO()
        original.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        recompressed = Image.open(buf).convert("RGB")
        diff = ImageChops.difference(original, recompressed)
        max_diff = max(v for _, v in diff.getextrema()) or 1
        return diff.point(lambda px: min(255, int(px * (255.0 / max_diff) * (scale / 10.0))))

    @staticmethod
    def _label_indices(model):
        raw = getattr(model.config, "id2label", {}) or {}
        labels = {int(k): str(v).lower() for k, v in raw.items()}
        forged = next((i for i, label in labels.items()
                       if any(x in label for x in ("forged", "forgery", "tampered", "tamper", "fake"))), None)
        real = next((i for i, label in labels.items()
                     if any(x in label for x in ("real", "genuine", "authentic", "bona"))), None)
        if forged is None and len(labels) >= 2:
            forged = 1
        if real is None and len(labels) >= 2:
            real = 0
        return real, forged

    def predict(self, path):
        processor, model = self._load()
        image = Image.open(path).convert("RGB")
        ela = self._ela(image)
        blended = Image.blend(image, ela, alpha=0.3)
        inputs = processor(images=blended, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
        real_idx, forged_idx = self._label_indices(model)
        if real_idx is None or forged_idx is None:
            raise RuntimeError("Document forgery model does not expose a binary real/forged label mapping.")
        real_probability = float(probs[real_idx].item())
        forged_probability = float(probs[forged_idx].item())
        label = "FORGED" if forged_probability >= real_probability else "REAL"
        return {
            "label": label,
            "forged_probability": forged_probability,
            "real_probability": real_probability,
            "confidence": max(real_probability, forged_probability),
            "model_version": MODEL_ID,
            "model_source": MODEL_ID,
            "analysis_scope": "Document forgery/tampering screening",
            "preprocessing": "RGB + ELA blend (alpha=0.3)",
            "interpretation": "AI-assisted document forgery screening; human verification is required for consequential decisions."
        }
