import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

MODEL_ID = 'Vansh180/deepfake-audio-wav2vec2'

class AudioDetector:
    _processor = None
    _model = None

    @classmethod
    def _load(cls):
        if cls._model is None:
            cls._processor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
            cls._model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
            cls._model.eval()
        return cls._processor, cls._model

    def predict(self, path):
        processor, model = self._load()
        wav, sr = torchaudio.load(path)
        wav = wav.mean(dim=0)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        # Use a bounded 4-second window to match the detector's training input.
        target = 16000 * 4
        if wav.numel() < target:
            wav = torch.nn.functional.pad(wav, (0, target - wav.numel()))
        else:
            wav = wav[:target]
        inputs = processor(wav.numpy(), sampling_rate=16000, return_tensors='pt')
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
        idx = int(probs.argmax())
        label = model.config.id2label.get(idx, str(idx)).lower()
        spoof = float(probs[idx].item()) if label in {'spoof', 'fake', 'deepfake'} else float(1.0 - probs[idx].item())
        return {
            'label': 'AI_GENERATED' if label in {'spoof', 'fake', 'deepfake'} else 'HUMAN',
            'fake_probability': spoof,
            'real_probability': 1.0 - spoof,
            'confidence': float(probs[idx].item()),
            'model_version': 'Vansh180/deepfake-audio-wav2vec2',
            'dataset': 'ASVspoof 2021 PA (model card reported)',
            'window_seconds': 4,
            'interpretation': 'AI-assisted speech-spoof screening; codec and attack-family limitations apply.'
        }
