import cv2
import torch
from PIL import Image

from ml.image.infer import ImageDetector


class VideoDetector:
    """
    Multi-frame video screening.

    Current architecture:
        video
          -> uniformly sampled frames
          -> image deepfake detector
          -> robust frame aggregation
          -> video-level decision

    This is NOT a temporal deepfake model.
    """

    def __init__(self, max_frames=24):
        self.detector = ImageDetector()
        self.max_frames = max_frames

    def predict(self, path):

        cap = cv2.VideoCapture(str(path))

        if not cap.isOpened():
            raise ValueError(
                "Unable to open video file."
            )

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        fps = (
            cap.get(cv2.CAP_PROP_FPS)
            or 25.0
        )

        if total_frames <= 0:
            cap.release()
            raise ValueError(
                "Video contains no decodable frames."
            )

        sample_count = min(
            self.max_frames,
            total_frames,
        )

        # Uniform sampling.
        indices = torch.linspace(
            0,
            total_frames - 1,
            sample_count,
        ).round().long().tolist()

        frame_results = []

        for frame_index in indices:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(frame_index),
            )

            ok, frame = cap.read()

            if not ok:
                continue

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            image = Image.fromarray(rgb)

            import tempfile

            with tempfile.NamedTemporaryFile(
                suffix=".jpg"
            ) as temporary:

                image.save(
                    temporary.name,
                    format="JPEG",
                    quality=95,
                )

                result = self.detector.predict(
                    temporary.name
                )

            probability = float(
                result["fake_probability"]
            )

            timestamp = (
                frame_index / fps
            )

            frame_results.append(
                {
                    "frame": int(frame_index),
                    "timestamp": round(
                        timestamp,
                        3,
                    ),
                    "fake_probability": round(
                        probability,
                        4,
                    ),
                }
            )

        cap.release()

        if not frame_results:
            raise ValueError(
                "No decodable sampled frames."
            )

        scores = torch.tensor(
            [
                x["fake_probability"]
                for x in frame_results
            ],
            dtype=torch.float32,
        )

        mean_score = float(
            scores.mean().item()
        )

        median_score = float(
            scores.median().item()
        )

        max_score = float(
            scores.max().item()
        )

        # Use top 25% rather than only one strongest frame.
        top_count = max(
            3,
            int(len(scores) * 0.25),
        )

        top_scores = torch.topk(
            scores,
            k=min(top_count, len(scores)),
        ).values

        top_mean = float(
            top_scores.mean().item()
        )

        # Count how consistently suspicious the video is.
        suspicious_threshold = 0.60

        suspicious_frames = int(
            (
                scores >= suspicious_threshold
            ).sum().item()
        )

        suspicious_ratio = (
            suspicious_frames / len(scores)
        )

        # More robust than:
        # "one frame > threshold = fake"
        ai_generated = (
            (
                top_mean >= 0.60
                and suspicious_ratio >= 0.20
            )
            or (
                median_score >= 0.55
                and suspicious_ratio >= 0.35
            )
        )

        label = (
            "AI_GENERATED"
            if ai_generated
            else "HUMAN"
        )

        if ai_generated:
            confidence = min(
                0.99,
                0.50
                + 0.30 * top_mean
                + 0.20 * suspicious_ratio,
            )
        else:
            confidence = min(
                0.99,
                0.50
                + 0.50 * (1.0 - median_score),
            )

        suspicious_timestamps = [
            x["timestamp"]
            for x in frame_results
            if x["fake_probability"]
            >= 0.75
        ]

        return {
            "label": label,

            "fake_probability": round(
                median_score,
                4,
            ),

            "real_probability": round(
                1.0 - median_score,
                4,
            ),

            "confidence": round(
                float(confidence),
                4,
            ),

            "model_version":
                "CommunityForensics-DeepfakeDet-ViT "
                "multi-frame ensemble",

            "sampled_frames":
                len(frame_results),

            "fps":
                round(float(fps), 3),

            "mean_fake_probability":
                round(mean_score, 4),

            "median_fake_probability":
                round(median_score, 4),

            "max_fake_probability":
                round(max_score, 4),

            "top_frame_mean":
                round(top_mean, 4),

            "suspicious_frame_ratio":
                round(suspicious_ratio, 4),

            "frame_scores":
                frame_results,

            "suspicious_timestamps":
                suspicious_timestamps,

            "decision": (
                "Consistent visual AI/deepfake indicators "
                "were detected across sampled frames."
                if ai_generated
                else
                "The sampled frames did not contain "
                "sufficiently consistent AI/deepfake indicators."
            ),

            "interpretation":
                "Multi-frame visual screening using an "
                "image detector. This implementation does "
                "not model temporal relationships between "
                "frames. Human forensic verification remains required.",
        }