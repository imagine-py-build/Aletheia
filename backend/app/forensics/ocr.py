from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET


def _tesseract_image(image):
    """OCR one PIL image. Raises the underlying error if Tesseract is unavailable."""
    import pytesseract
    return pytesseract.image_to_string(image)


def _ocr_pdf(path: Path):
    """
    Handle both normal text PDFs and scanned/image PDFs.

    First extract embedded PDF text directly. If a page has little/no text,
    render that page to an image and run Tesseract OCR on the rendered page.
    This avoids passing a PDF file directly to PIL.Image.open(), which is
    what caused the previous 'cannot identify image file' error.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(str(path))
    page_texts = []
    ocr_pages = 0

    try:
        for page_number, page in enumerate(doc, start=1):
            text = (page.get_text("text") or "").strip()

            # Scanned/image-only pages need raster OCR.
            if len(text) < 10:
                pix = page.get_pixmap(
                    matrix=fitz.Matrix(2.0, 2.0),
                    alpha=False
                )
                from PIL import Image
                import io

                image = Image.open(io.BytesIO(pix.tobytes("png")))
                text = _tesseract_image(image).strip()
                if text:
                    ocr_pages += 1

            if text:
                page_texts.append(f"[Page {page_number}]\n{text}")

        return {
            "text": "\n\n".join(page_texts),
            "engine": "pymupdf+tesseract",
            "pages": len(doc),
            "ocr_pages": ocr_pages,
        }
    finally:
        doc.close()


def _extract_docx_text(path: Path):
    """Extract text from DOCX without adding another Python dependency."""
    with zipfile.ZipFile(path, "r") as zf:
        xml = zf.read("word/document.xml")

    root = ET.fromstring(xml)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

    paragraphs = []
    for paragraph in root.findall(".//w:p", ns):
        parts = []
        for node in paragraph.findall(".//w:t", ns):
            if node.text:
                parts.append(node.text)
        text = "".join(parts).strip()
        if text:
            paragraphs.append(text)

    return {
        "text": "\n".join(paragraphs),
        "engine": "docx-xml",
    }


def extract_text(path):
    """
    Genuine OCR/text extraction for images and common document formats.

    PDF handling is explicit because PIL cannot open PDF files as images.
    DOCX/TXT are extracted as documents; image formats use Tesseract.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    try:
        if suffix == ".pdf":
            return _ocr_pdf(path)

        if suffix == ".docx":
            return _extract_docx_text(path)

        if suffix in {".txt", ".md", ".csv"}:
            return {
                "text": path.read_text(encoding="utf-8", errors="replace"),
                "engine": "text-extraction",
            }

        # Image OCR fallback.
        from PIL import Image
        with Image.open(path) as image:
            text = _tesseract_image(image)

        return {
            "text": text,
            "engine": "tesseract",
        }

    except Exception as e:
        return {
            "text": "",
            "engine": "unavailable",
            "error": str(e),
        }
