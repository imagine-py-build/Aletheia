import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


MODEL_ID = "buildborderless/CommunityForensics-DeepfakeDet-ViT"


class ImageDetector:

    _processor = None
    _model = None

    @classmethod
    def _load(cls):

        if cls._model is None:

            print("Loading CommunityForensics model...")

            cls._processor = AutoImageProcessor.from_pretrained(
                MODEL_ID,
                force_download=True
            )

            cls._model = AutoModelForImageClassification.from_pretrained(
                MODEL_ID,
                force_download=True
            )

            cls._model.eval()

            # IMPORTANT CHECK
            print("Number of labels:", cls._model.config.num_labels)

            if cls._model.config.num_labels != 1:
                raise RuntimeError(
                    "Wrong/old CommunityForensics model configuration. "
                    "Expected num_labels=1."
                )

        return cls._processor, cls._model

    def predict(self, path):

        processor, model = self._load()

        # Load image
        image = Image.open(path).convert("RGB")

        # Preprocess image
        inputs = processor(
            images=image,
            return_tensors="pt"
        )

        # Model prediction
        with torch.no_grad():

            outputs = model(**inputs)

            logit = outputs.logits.reshape(-1)[0]

            fake_probability = torch.sigmoid(logit).item()

        # Real probability
        real_probability = 1.0 - fake_probability

        # Decision
        if fake_probability >= 0.5:
            label = "FAKE"
            confidence = fake_probability
        else:
            label = "REAL"
            confidence = real_probability

        return {

            "label": label,

            "fake_probability": fake_probability,

            "real_probability": real_probability,

            "confidence": confidence,

            "model_version":
                "CommunityForensics-DeepfakeDet-ViT",

            "model_source":
                MODEL_ID,

            "input_mode":
                "full image",

            "interpretation":
                "AI-assisted screening result; human verification is required."
        }