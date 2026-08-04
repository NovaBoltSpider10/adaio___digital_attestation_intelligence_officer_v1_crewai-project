# utils/ocr.py

import os
import logging
import base64
import requests
import json
from typing import Union, Dict, Any
from PIL import Image

try:
    import pypdf
except ImportError:
    import PyPDF2 as pypdf

logger = logging.getLogger("ADAIO")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "gemma3")


def ocr_with_ollama(image_path_or_pil: Union[str, Image.Image]) -> Dict[str, Any]:
    """
    Sends an image to Ollama to extract text AND perform document quality assessment.
    Returns a structured dictionary containing raw text and quality metrics.
    """
    try:
        if isinstance(image_path_or_pil, str):
            with open(image_path_or_pil, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        else:
            import io
            buffered = io.BytesIO()
            image_path_or_pil.save(buffered, format="PNG")
            base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

        prompt = (
            "Analyze this document image. "
            "1. Extract all legible text.\n"
            "2. Assess visual document quality (legibility: high, medium, low).\n"
            "3. Identify any quality flags or visual anomalies (e.g., blurriness, cropping, low resolution, potential tampering, missing seals).\n\n"
            "Respond strictly in valid JSON format with the following keys:\n"
            "{\n"
            '  "extracted_text": "full text transcribed here",\n'
            '  "quality_assessment": {\n'
            '    "legibility": "high|medium|low",\n'
            '    "flags": ["flag_1", "flag_2"]\n'
            "  }\n"
            "}"
        )

        payload = {
            "model": OLLAMA_VISION_MODEL,
            "prompt": prompt,
            "images": [base64_image],
            "format": "json",
            "stream": False
        }

        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()

        result = response.json().get("response", "{}")
        parsed = json.loads(result)
        
        return {
            "text": parsed.get("extracted_text", "").strip(),
            "quality_assessment": parsed.get("quality_assessment", {"legibility": "medium", "flags": []})
        }

    except Exception as e:
        logger.error(f"Ollama OCR failed: {e}")
        return {
            "text": f"[OCR ERROR: {str(e)}]",
            "quality_assessment": {
                "legibility": "low",
                "flags": [f"OCR_PROCESSING_ERROR: {str(e)}"]
            }
        }


def extract_file_content(file_path: str) -> Dict[str, Any]:
    """Utility function to extract text content and quality assessment from files."""
    if not os.path.exists(file_path):
        return {
            "text": "ERROR: File not found.",
            "quality_assessment": {"legibility": "low", "flags": ["FILE_NOT_FOUND"]}
        }

    ext = os.path.splitext(file_path)[1].lower()

    # 1. Image Files -> OCR with Quality Assessment
    if ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp']:
        return ocr_with_ollama(file_path)

    # 2. PDF Files -> Digital extraction + quality checks
    elif ext == '.pdf':
        text = ""
        flags = []
        try:
            reader = pypdf.PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            logger.warning(f"Digital PDF extraction error on {file_path}: {e}")
            flags.append("DIGITAL_EXTRACTION_FAILED")

        if text.strip():
            return {
                "text": text.strip(),
                "quality_assessment": {
                    "legibility": "high",
                    "flags": flags
                }
            }

        # Scanned PDF fallback
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(file_path)
            ocr_text = ""
            combined_flags = []
            
            for img in images:
                res = ocr_with_ollama(img)
                ocr_text += res["text"] + "\n"
                combined_flags.extend(res["quality_assessment"].get("flags", []))

            return {
                "text": ocr_text.strip() if ocr_text.strip() else "[OCR WARNING: No legible text detected]",
                "quality_assessment": {
                    "legibility": "medium" if ocr_text.strip() else "low",
                    "flags": list(set(combined_flags))
                }
            }
        except Exception as e:
            return {
                "text": f"ERROR extracting PDF: {str(e)}",
                "quality_assessment": {"legibility": "low", "flags": ["PDF_OCR_FALLBACK_FAILED"]}
            }

    # 3. Plain text files
    else:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                return {
                    "text": content,
                    "quality_assessment": {
                        "legibility": "high" if content.strip() else "low",
                        "flags": [] if content.strip() else ["EMPTY_TEXT_FILE"]
                    }
                }
        except Exception as e:
            return {
                "text": f"ERROR reading file: {str(e)}",
                "quality_assessment": {"legibility": "low", "flags": ["READ_ERROR"]}
            }