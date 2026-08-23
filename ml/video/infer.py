import cv2
import math
import tempfile
from statistics import mean, median

from PIL import Image

from ml.image.infer import ImageDetector


class VideoDetector:
    """
    Multi-frame video deepfake screening using the trained image detector.

    Important:
    This is a frame-ensemble detector, not a trained temporal video model.
    It improves the previous implementation by:
      - sampling more frames when possible
      - avoiding duplicate frame indexes
      - using robust top-frame aggregation instead of a plain mean
      - retaining frame-level evidence and suspicious timestamps
      - returning only HUMAN or AI-GENERATED as the final label

    The final result is still an AI-assisted screening result and should not
    be treated as legal certainty.
    """

    def __init__(self, max_frames=32, suspicious_threshold=0.65):
        self.detector = ImageDetector()
        self.max_frames = max_frames
        self.suspicious_threshold = suspicious_threshold

    @staticmethod
    def _sample_indexes(total_frames, max_frames):
        """Return evenly distributed, unique frame indexes."""
        if total_frames <= 0:
            return []

        count = min(max_frames, total_frames)

        if count == 1:
            return [0]

        # Integer interpolation without torch so there are no accidental
        # duplicate indexes caused by rounding.
        indexes = []
        for i in range(count):
            idx = round(i * (total_frames - 1) / (count - 1))
            if not indexes or idx != indexes[-1]:
                indexes.append(idx)

        return indexes

    @staticmethod
    def _safe_probability(value):
        """Convert detector output to a valid probability."""
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0

        return max(0.0, min(1.0, value))

    @staticmethod
    def _aggregate(values):
        """
        Robust aggregation.

        A plain mean can hide a deepfake when only some frames contain strong
        manipulation evidence. We therefore combine:
          - overall mean
          - median
          - mean of the strongest 25% of frames
          - 75th percentile

        The top-frame component gives intermittent AI artifacts more influence
        while the mean/median components prevent a single bad frame from
        automatically deciding the whole video.
        """
        if not values:
            raise ValueError("No frame scores available")

        ordered = sorted(values)
        n = len(ordered)

        overall_mean = mean(ordered)
        middle = median(ordered)

        top_count = max(1, math.ceil(n * 0.25))
        top_mean = mean(ordered[-top_count:])

        # Linear interpolation for the 75th percentile.
        if n == 1:
            q75 = ordered[0]
        else:
            position = 0.75 * (n - 1)
            lower = math.floor(position)
            upper = math.ceil(position)

            if lower == upper:
                q75 = ordered[lower]
            else:
                fraction = position - lower
                q75 = (
                    ordered[lower] * (1.0 - fraction)
                    + ordered[upper] * fraction
                )

        # Robust ensemble score.
        #
        # Top-frame evidence receives the greatest weight because AI video
        # artifacts can be intermittent after social-media recompression.
        score = (
            0.45 * top_mean
            + 0.25 * q75
            + 0.20 * overall_mean
            + 0.10 * middle
        )

        return {
            "aggregate": max(0.0, min(1.0, score)),
            "mean": overall_mean,
            "median": middle,
            "top_25_percent_mean": top_mean,
            "q75": q75,
            "top_frame_count": top_count,
        }

    def predict(self, path):
        cap = cv2.VideoCapture(str(path))

        if not cap.isOpened():
            raise ValueError("Unable to open video")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)

        if total <= 0:
            cap.release()
            raise ValueError("No decodable video frames")

        indexes = self._sample_indexes(total, self.max_frames)
        frame_scores = []

        for idx in indexes:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()

            if not ok or frame is None:
                continue

            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)

                # ImageDetector currently accepts a file path, so use a
                # temporary JPEG for each sampled frame.
                with tempfile.NamedTemporaryFile(
                    suffix=".jpg", delete=True
                ) as tmp:
                    image.save(tmp.name, format="JPEG", quality=95)
                    out = self.detector.predict(tmp.name)

                fake_probability = self._safe_probability(
                    out.get("fake_probability", 0.0)
                )

                frame_scores.append(
                    {
                        "frame": int(idx),
                        "timestamp": round(idx / fps, 3),
                        "fake_probability": round(fake_probability, 6),
                    }
                )

            except Exception:
                # A bad individual frame should not destroy the entire
                # video analysis.
                continue

        cap.release()

        if not frame_scores:
            raise ValueError("No decodable sampled frames")

        values = [x["fake_probability"] for x in frame_scores]
        stats = self._aggregate(values)
        aggregate = stats["aggregate"]

        # Mark frames that contain meaningful suspicious evidence.
        suspicious = [
            x
            for x in frame_scores
            if x["fake_probability"] >= self.suspicious_threshold
        ]

        # Strongest frames first for investigator review.
        strongest = sorted(
            frame_scores,
            key=lambda x: x["fake_probability"],
            reverse=True,
        )[: min(8, len(frame_scores))]

        # Final binary decision.
        #
        # 0.50 is the midpoint of the model probability. The robust
        # aggregation above prevents the old plain-average calculation from
        # washing out intermittent strong evidence.
        label = "AI-GENERATED" if aggregate >= 0.50 else "HUMAN"

        return {
            "label": label,
            "fake_probability": round(aggregate, 6),
            "real_probability": round(1.0 - aggregate, 6),
            "confidence": round(max(aggregate, 1.0 - aggregate), 6),

            "model_version": (
                "CommunityForensics-DeepfakeDet-ViT "
                "robust frame ensemble"
            ),

            "sampled_frames": len(frame_scores),
            "total_video_frames": total,
            "fps": round(fps, 3),

            # Aggregation diagnostics make the result auditable.
            "aggregation": {
                "method": "robust_top_quartile_ensemble",
                "mean": round(stats["mean"], 6),
                "median": round(stats["median"], 6),
                "top_25_percent_mean": round(
                    stats["top_25_percent_mean"], 6
                ),
                "q75": round(stats["q75"], 6),
                "top_frame_count": stats["top_frame_count"],
            },

            "frame_scores": frame_scores,

            "suspicious_timestamps": [
                x["timestamp"] for x in suspicious
            ],

            "strongest_frames": strongest,

            "interpretation": (
                "AI-assisted multi-frame visual screening. "
                "The result combines the overall frame distribution with "
                "the strongest 25% of frames so intermittent manipulation "
                "evidence is not hidden by social-media recompression. "
                "This remains a frame-ensemble method, not a trained "
                "temporal deepfake model, and should not be treated as "
                "legal certainty."
            ),
        }