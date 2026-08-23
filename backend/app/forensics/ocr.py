def extract_text(path):
    # PaddleOCR is optional because it has a heavier install footprint. If unavailable,
    # use pytesseract as a genuine OCR fallback rather than synthetic text.
    try:
        from paddleocr import PaddleOCR
        ocr=PaddleOCR(use_doc_orientation_classify=False,use_doc_unwarping=False,use_textline_orientation=False,lang='en')
        result=ocr.predict(str(path))
        texts=[]
        for page in result:
            data=page.get('rec_texts',[]) if isinstance(page,dict) else []
            texts.extend(data)
        return {'text':'\n'.join(texts),'engine':'paddleocr'}
    except Exception:
        try:
            import pytesseract
            from PIL import Image
            return {'text':pytesseract.image_to_string(Image.open(path)),'engine':'tesseract'}
        except Exception as e:
            return {'text':'','engine':'unavailable','error':str(e)}
