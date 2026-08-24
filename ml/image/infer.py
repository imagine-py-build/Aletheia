from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)

MODEL_ID = "buildborderless/CommunityForensics-DeepfakeDet-ViT"

# Conservative decision threshold.
# This should eventually be calibrated against your evaluation dataset.
FAKE_THRESHOLD = 0.30


class ImageDetector:
    """
    AI-generated image screening using CommunityForensics DeepfakeDet-ViT.

    Strategy:
        1. Analyze the complete image.
        2. Detect faces if present.
        3. Analyze the largest face separately.
        4. Combine the signals instead of allowing the face crop
           to completely replace the original image.

    Output is intentionally binary:
        FAKE
        REAL

    This is an AI-assisted forensic screening result, not proof
    of authenticity or identity.
    """

    _processor = None
    _model = None

    @classmethod
    def _load(cls):
        if cls._model is None:
            cls._processor = (
                AutoImageProcessor.from_pretrained(MODEL_ID)
            )

            cls._model = (
                AutoModelForImageClassification.from_pretrained(
                    MODEL_ID
                )
            )

            cls._model.eval()

        return cls._processor, cls._model

    @staticmethod
    def _largest_face(
        image: Image.Image,
    ):
        """
        Detect the largest face without making the face crop
        the primary image input.
        """

        image_np = np.array(image)

        bgr = cv2.cvtColor(
            image_np,
            cv2.COLOR_RGB2BGR,
        )

        gray = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2GRAY,
        )

        cascade_path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        detector = cv2.CascadeClassifier(
            cascade_path
        )

        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=5,
            minSize=(64, 64),
        )

        if len(faces) == 0:
            return None, {
                "face_detected": False,
                "face_count": 0,
            }

        x, y, w, h = max(
            faces,
            key=lambda f: int(f[2]) * int(f[3]),
        )

        pad = int(
            max(w, h) * 0.20
        )

        x0 = max(
            0,
            x - pad,
        )

        y0 = max(
            0,
            y - pad,
        )

        x1 = min(
            image.width,
            x + w + pad,
        )

        y1 = min(
            image.height,
            y + h + pad,
        )

        crop = image.crop(
            (
                x0,
                y0,
                x1,
                y1,
            )
        )

        return crop, {
            "face_detected": True,
            "face_count": len(faces),
            "face_box": [
                int(x0),
                int(y0),
                int(x1),
                int(y1),
            ],
        }

    @staticmethod
    def _center_crop(image):
        """
        Create a central crop so that the model receives
        another view of the image without relying on a face.
        """

        width, height = image.size

        crop_size = min(
            width,
            height,
        )

        left = int(
            (width - crop_size) / 2
        )

        top = int(
            (height - crop_size) / 2
        )

        return image.crop(
            (
                left,
                top,
                left + crop_size,
                top + crop_size,
            )
        )

    def _model_probability(
        self,
        image,
    ):
        """
        Obtain fake probability while respecting the
        model's output structure.
        """

        processor, model = self._load()

        inputs = processor(
            images=image,
            return_tensors="pt",
        )

        with torch.inference_mode():
            output = model(**inputs)

        logits = output.logits

        # Binary classifier with two output logits.
        if logits.ndim == 2 and logits.shape[-1] == 2:

            probabilities = torch.softmax(
                logits,
                dim=-1,
            )[0]

            labels = (
                getattr(
                    model.config,
                    "id2label",
                    {},
                )
                or {}
            )

            fake_index = None

            for key, value in labels.items():

                label = str(value).lower()

                if any(
                    token in label
                    for token in (
                        "fake",
                        "deepfake",
                        "synthetic",
                        "forged",
                        "ai",
                    )
                ):
                    fake_index = int(key)
                    break

            if fake_index is None:
                fake_index = 1

            return float(
                probabilities[
                    fake_index
                ].item()
            )

        # Single-logit binary classifier.
        if logits.ndim == 2 and logits.shape[-1] == 1:

            return float(
                torch.sigmoid(
                    logits[0, 0]
                ).item()
            )

        # Fallback for unusual checkpoint output.
        flat = logits.reshape(-1)

        if flat.numel() == 1:

            return float(
                torch.sigmoid(
                    flat[0]
                ).item()
            )

        raise RuntimeError(
            "Unsupported image model output shape: "
            f"{tuple(logits.shape)}"
        )

    def predict(self, path):

        image = Image.open(
            path
        ).convert("RGB")

        # -------------------------------------------------
        # 1. FULL IMAGE
        # -------------------------------------------------

        full_probability = (
            self._model_probability(
                image
            )
        )

        # -------------------------------------------------
        # 2. CENTER CROP
        # -------------------------------------------------

        center = self._center_crop(
            image
        )

        center_probability = (
            self._model_probability(
                center
            )
        )

        # -------------------------------------------------
        # 3. FACE CROP
        # -------------------------------------------------

        face_crop, face_info = (
            self._largest_face(
                image
            )
        )

        face_probability = None

        if face_crop is not None:

            face_probability = (
                self._model_probability(
                    face_crop
                )
            )

        # -------------------------------------------------
        # 4. ROBUST AGGREGATION
        # -------------------------------------------------

        probabilities = [
            full_probability,
            center_probability,
        ]

        if face_probability is not None:
            probabilities.append(
                face_probability
            )

        scores = np.asarray(
            probabilities,
            dtype=np.float32,
        )

        # Median is deliberately used instead of maximum.
        #
        # Why?
        #
        # One anomalous crop should not automatically
        # classify the complete image as AI-generated.
        median_probability = float(
            np.median(scores)
        )

        mean_probability = float(
            np.mean(scores)
        )

        max_probability = float(
            np.max(scores)
        )

        # -------------------------------------------------
        # 5. FINAL DECISION
        # -------------------------------------------------

        fake = (
            median_probability
            >= FAKE_THRESHOLD
        )

        label = (
            "FAKE"
            if fake
            else "REAL"
        )

        if fake:

            confidence = (
                median_probability
            )

        else:

            confidence = (
                1.0
                - median_probability
            )

        return {

            "label": label,

            "fake_probability": round(
                median_probability,
                4,
            ),

            "real_probability": round(
                1.0 - median_probability,
                4,
            ),

            "confidence": round(
                float(confidence),
                4,
            ),

            "model_version":
                "CommunityForensics-DeepfakeDet-ViT",

            "model_source":
                MODEL_ID,

            "full_image_probability":
                round(
                    full_probability,
                    4,
                ),

            "center_crop_probability":
                round(
                    center_probability,
                    4,
                ),

            "face_probability":
                (
                    round(
                        face_probability,
                        4,
                    )
                    if face_probability
                    is not None
                    else None
                ),

            "mean_probability":
                round(
                    mean_probability,
                    4,
                ),

            "max_probability":
                round(
                    max_probability,
                    4,
                ),

            "input_mode":
                "full image + center crop + face crop"
                if face_probability is not None
                else
                "full image + center crop",

            **face_info,

            "decision_threshold":
                FAKE_THRESHOLD,

            "interpretation":
                (
                    "AI-assisted image authenticity "
                    "screening using multiple image views. "
                    "The result is not proof of authenticity "
                    "and should be reviewed by a human "
                    "for consequential decisions."
                ),
        }