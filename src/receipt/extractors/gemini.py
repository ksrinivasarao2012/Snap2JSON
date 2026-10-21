"""Gemini backend (google-genai): one call = prompt + optional OCR text or image -> JSON text.

The model name comes from the GEMINI_MODEL env var (set it in .env after checking Google AI
Studio for the current free-tier Flash model). The key comes from GEMINI_API_KEY.
"""
import io
import os

from dotenv import load_dotenv

from receipt.schema import Receipt

MAX_SIDE = 2048  # downscale big photos before upload (keeps requests small)


def prepare_image(path) -> bytes:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class GeminiBackend:
    def __init__(self, model: str = None):
        load_dotenv()
        self.model = model or os.environ.get("GEMINI_MODEL")
        if not self.model:
            raise RuntimeError("Set GEMINI_MODEL in .env (the exact Gemini Flash model name).")
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY in .env.")
        from google import genai

        self._client = genai.Client(api_key=key)

    def generate(self, prompt: str, text: str = None, image_png: bytes = None) -> dict:
        from google.genai import types

        contents = [prompt]
        if text is not None:
            contents.append("Receipt OCR text:\n" + text)
        if image_png is not None:
            contents.append(types.Part.from_bytes(data=image_png, mime_type="image/png"))
        resp = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0, response_mime_type="application/json", response_schema=Receipt
            ),
        )
        usage = resp.usage_metadata
        return {
            "text": resp.text or "",
            "input_tokens": getattr(usage, "prompt_token_count", None),
            "output_tokens": getattr(usage, "candidates_token_count", None),
        }
