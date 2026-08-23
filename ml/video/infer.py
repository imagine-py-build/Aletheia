import cv2
import torch
from PIL import Image
from ml.image.infer import ImageDetector

class VideoDetector:
    """Multi-frame ensemble using the real image deepfake detector.

    This intentionally does not pretend to be a temporal model. It samples
    multiple frames, scores each with the same trained detector, and reports
    the aggregate plus suspicious timestamps.
    """
    def __init__(self, max_frames=24):
        self.detector = ImageDetector()
        self.max_frames = max_frames

    def predict(self, path):
        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if not total:
            cap.release()
            raise ValueError('No decodable video frames')
        idxs = [int(i) for i in torch.linspace(0, max(total-1, 0), min(self.max_frames, total))]
        frame_scores = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            tmp = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # The detector accepts a path, so use an in-memory temporary file.
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg') as f:
                tmp.save(f.name, format='JPEG', quality=95)
                out = self.detector.predict(f.name)
            frame_scores.append({'frame': idx, 'timestamp': round(idx / fps, 3), 'fake_probability': out['fake_probability']})
        cap.release()
        if not frame_scores:
            raise ValueError('No decodable sampled frames')
        values = [x['fake_probability'] for x in frame_scores]
        aggregate = sum(values) / len(values)
        suspicious = [x for x in frame_scores if x['fake_probability'] >= 0.75]
        return {
            'label': 'FAKE' if aggregate >= 0.5 else 'REAL',
            'fake_probability': aggregate,
            'real_probability': 1.0 - aggregate,
            'confidence': max(aggregate, 1.0 - aggregate),
            'model_version': 'CommunityForensics-DeepfakeDet-ViT frame ensemble',
            'sampled_frames': len(frame_scores),
            'fps': fps,
            'frame_scores': frame_scores,
            'suspicious_timestamps': [x['timestamp'] for x in suspicious],
            'interpretation': 'Multi-frame visual screening. It is not a temporal deepfake model and should not be treated as legal certainty.'
        }
