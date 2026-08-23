from pathlib import Path
import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_ID = 'buildborderless/CommunityForensics-DeepfakeDet-ViT'

class ImageDetector:
    """Real pretrained deepfake detector, downloaded and cached on first use.

    CommunityForensics DeepfakeDet-ViT is a ViT-Small detector trained for
    AI-generated image detection. It emits one fake logit; sigmoid converts it
    to a fake probability. This is a research screening result, not legal proof.
    """
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
    def _largest_face(image: Image.Image) -> tuple[Image.Image, dict]:
        bgr = cv2.cvtColor(__import__('numpy').array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        cascade = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        detector = cv2.CascadeClassifier(cascade)
        faces = detector.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(64, 64))
        if len(faces) == 0:
            return image, {'face_detected': False, 'face_count': 0}
        x, y, w, h = max(faces, key=lambda f: int(f[2]) * int(f[3]))
        pad = int(max(w, h) * 0.20)
        x0, y0 = max(0, x-pad), max(0, y-pad)
        x1, y1 = min(image.width, x+w+pad), min(image.height, y+h+pad)
        return image.crop((x0, y0, x1, y1)), {'face_detected': True, 'face_count': len(faces), 'face_box': [x0, y0, x1, y1]}

    def predict(self, path):
        processor, model = self._load()
        image = Image.open(path).convert('RGB')
        crop, face_info = self._largest_face(image)
        inputs = processor(images=crop, return_tensors='pt')
        with torch.no_grad():
            logit = model(**inputs).logits.reshape(-1)[0]
            fake_probability = float(torch.sigmoid(logit).item())
        label = 'FAKE' if fake_probability >= 0.5 else 'REAL'
        confidence = fake_probability if label == 'FAKE' else 1.0 - fake_probability
        return {
            'label': label,
            'fake_probability': fake_probability,
            'real_probability': 1.0 - fake_probability,
            'confidence': confidence,
            'model_version': 'CommunityForensics-DeepfakeDet-ViT',
            'model_source': 'buildborderless/CommunityForensics-DeepfakeDet-ViT',
            'input_mode': 'largest-face crop' if face_info['face_detected'] else 'full image (no face detected)',
            **face_info,
            'interpretation': 'AI-assisted screening result; human verification is required for consequential decisions.'
        }
