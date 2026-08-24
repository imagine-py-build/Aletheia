import io
from pathlib import Path

import torch
from PIL import Image, ImageChops

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)


MODEL_ID = "zodumair/document-forgery-detector"


class DocumentForgeryDetector:
    """
    Document forgery/tampering screening detector.

    Supports:
      - JPG
      - JPEG
      - PNG
      - WEBP
      - BMP
      - TIFF
      - PDF

    The classifier is binary at application level:

        AI_GENERATED
        REAL

    For PDFs, pages are rendered and analyzed individually.
    """

    _processor = None
    _model = None

    @classmethod
    def _load(cls):
        if cls._model is None:
            cls._processor = (
                AutoImageProcessor.from_pretrained(
                    MODEL_ID
                )
            )

            cls._model = (
                AutoModelForImageClassification
                .from_pretrained(MODEL_ID)
            )

            cls._model.eval()

        return cls._processor, cls._model

    @staticmethod
    def _ela(
        image: Image.Image,
        quality=90,
        scale=15.0,
    ):
        """
        Error Level Analysis.

        Recompresses the image and highlights differences.
        This is an additional forensic preprocessing signal,
        not proof of manipulation by itself.
        """

        original = image.convert("RGB")

        buf = io.BytesIO()

        original.save(
            buf,
            format="JPEG",
            quality=quality,
        )

        buf.seek(0)

        recompressed = (
            Image.open(buf)
            .convert("RGB")
        )

        diff = ImageChops.difference(
            original,
            recompressed,
        )

        max_diff = (
            max(
                value
                for _, value
                in diff.getextrema()
            )
            or 1
        )

        return diff.point(
            lambda px: min(
                255,
                int(
                    px
                    * (255.0 / max_diff)
                    * (scale / 10.0)
                ),
            )
        )

    @staticmethod
    def _label_indices(model):
        raw = (
            getattr(
                model.config,
                "id2label",
                {},
            )
            or {}
        )

        labels = {
            int(k): str(v).lower()
            for k, v in raw.items()
        }

        forged = None
        real = None

        for index, label in labels.items():

            if any(
                word in label
                for word in (
                    "forged",
                    "forgery",
                    "tampered",
                    "tamper",
                    "fake",
                    "manipulated",
                    "fraud",
                )
            ):
                forged = index

            if any(
                word in label
                for word in (
                    "real",
                    "genuine",
                    "authentic",
                    "bona",
                )
            ):
                real = index

        # Fallback for binary checkpoints.
        if len(labels) >= 2:

            if forged is None:
                forged = 1

            if real is None:
                real = 0

        return real, forged

    @staticmethod
    def _load_pages(path):
        """
        Return a list of PIL images.

        Images produce one page.

        PDFs are rendered page-by-page using PyMuPDF.
        """

        suffix = Path(path).suffix.lower()

        if suffix != ".pdf":

            image = (
                Image.open(path)
                .convert("RGB")
            )

            return [image]

        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError(
                "PDF document analysis requires PyMuPDF. "
                "Install it with: pip install pymupdf"
            ) from exc

        document = fitz.open(path)

        pages = []

        try:
            for page_index in range(
                len(document)
            ):

                page = document[
                    page_index
                ]

                # 2x rendering gives the vision model
                # substantially more useful visual information.
                matrix = fitz.Matrix(
                    2.0,
                    2.0,
                )

                pixmap = page.get_pixmap(
                    matrix=matrix,
                    alpha=False,
                )

                image = Image.frombytes(
                    "RGB",
                    [
                        pixmap.width,
                        pixmap.height,
                    ],
                    pixmap.samples,
                )

                pages.append(image)

        finally:
            document.close()

        if not pages:
            raise ValueError(
                "PDF contains no renderable pages."
            )

        return pages

    def _predict_page(
        self,
        image,
        processor,
        model,
    ):
        """
        Analyze one rendered document page.
        """

        ela = self._ela(image)

        # Combine the original document with ELA information.
        blended = Image.blend(
            image,
            ela,
            alpha=0.30,
        )

        inputs = processor(
            images=blended,
            return_tensors="pt",
        )

        with torch.inference_mode():

            logits = model(
                **inputs
            ).logits[0]

            probs = torch.softmax(
                logits,
                dim=-1,
            )

        real_idx, forged_idx = (
            self._label_indices(model)
        )

        if (
            real_idx is None
            or forged_idx is None
        ):
            raise RuntimeError(
                "Document forgery model does not expose "
                "a binary real/forged label mapping."
            )

        real_probability = float(
            probs[real_idx].item()
        )

        forged_probability = float(
            probs[forged_idx].item()
        )

        return (
            real_probability,
            forged_probability,
        )

    def predict(self, path):

        processor, model = self._load()

        pages = self._load_pages(path)

        page_results = []

        for page_number, image in enumerate(
            pages,
            start=1,
        ):

            real_probability, forged_probability = (
                self._predict_page(
                    image,
                    processor,
                    model,
                )
            )

            page_results.append(
                {
                    "page": page_number,
                    "real_probability": round(
                        real_probability,
                        4,
                    ),
                    "forged_probability": round(
                        forged_probability,
                        4,
                    ),
                }
            )

        if not page_results:
            raise ValueError(
                "No document pages could be analyzed."
            )

        # ---------------------------------------------------------
        # DOCUMENT-LEVEL AGGREGATION
        # ---------------------------------------------------------

        forged_scores = [
            x["forged_probability"]
            for x in page_results
        ]

        real_scores = [
            x["real_probability"]
            for x in page_results
        ]

        average_forged = (
            sum(forged_scores)
            / len(forged_scores)
        )

        average_real = (
            sum(real_scores)
            / len(real_scores)
        )

        maximum_forged = max(
            forged_scores
        )

        # A document should not be marked AI-generated from
        # one mildly suspicious page.
        #
        # Require either:
        #
        # 1. Strong document-level evidence
        # OR
        #
        # 2. More than one strongly suspicious page.
        #
        strong_pages = sum(
            1
            for score in forged_scores
            if score >= 0.80
        )

        AI_DOCUMENT_THRESHOLD = 0.80
        STRONG_PAGE_THRESHOLD = 0.90

        highly_suspicious_pages = sum(
            1
            for score in forged_scores
            if score >= STRONG_PAGE_THRESHOLD
        )

        ai_generated = (
            average_forged
            >= AI_DOCUMENT_THRESHOLD
            or
            highly_suspicious_pages >= 2
        )

        label = (
            "AI_GENERATED"
            if ai_generated
            else "REAL"
        )

        # Use document-level probability rather than
        # automatically using the maximum page probability.
        if ai_generated:

            decision_probability = max(
                average_forged,
                maximum_forged,
            )

            confidence = min(
                0.99,
                0.50
                + 0.50 * decision_probability,
            )

            decision = (
                "Document-level visual evidence "
                "supports an AI-generated/forged classification."
            )

        else:

            decision_probability = average_forged

            confidence = min(
                0.99,
                0.50
                + 0.50 * (
                    1.0
                    - average_forged
                ),
            )

            decision = (
                "The analyzed document did not meet "
                "the AI-generated/forgery threshold."
            )

        suffix = (
            Path(path)
            .suffix
            .lower()
        )

        return {
            "label": label,

            "fake_probability": round(
                float(decision_probability),
                4,
            ),

            "forged_probability": round(
                float(decision_probability),
                4,
            ),

            "real_probability": round(
                float(
                    1.0
                    - decision_probability
                ),
                4,
            ),

            "confidence": round(
                float(confidence),
                4,
            ),

            "model_version": MODEL_ID,

            "model_source": MODEL_ID,

            "document_type": (
                "PDF"
                if suffix == ".pdf"
                else "IMAGE"
            ),

            "pages_analyzed": len(
                page_results
            ),

            "page_results": page_results,

            "strong_suspicious_pages": strong_pages,

            "highly_suspicious_pages": (
                highly_suspicious_pages
            ),

            "decision_thresholds": {
                "document_ai_threshold": (
                    AI_DOCUMENT_THRESHOLD
                ),
                "strong_page_threshold": (
                    STRONG_PAGE_THRESHOLD
                ),
            },

            "analysis_scope": (
                "Document visual forgery/tampering "
                "and AI-generated-content screening"
            ),

            "preprocessing": (
                "RGB rendering + Error Level Analysis "
                "blend (alpha=0.3)"
            ),

            "decision": decision,

            "interpretation": (
                "Binary AI-generated versus REAL "
                "document screening using a pretrained "
                "document-forgery image classifier. "
                "PDF pages are rendered and analyzed "
                "individually before document-level "
                "aggregation. The result is an AI-assisted "
                "screening finding and requires human "
                "verification for consequential decisions."
            ),
        }