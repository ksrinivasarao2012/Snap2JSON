"""Qwen3-VL backend (Hugging Face transformers): one call = prompt + optional OCR text or image -> text.

Runs on the lab GPU (needs the `models` extra). Same interface as GeminiBackend, so the same
OcrLlm / Vlm extractors work with it. Greedy decoding (no sampling) for repeatable output.
Not tested here: only the fake-backend tests run without a GPU.
"""
import io

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"  # provisional; pass --model for another size
MAX_NEW_TOKENS = 1024


class QwenBackend:
    def __init__(self, model: str = DEFAULT_MODEL):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model = model
        self._proc = AutoProcessor.from_pretrained(model)
        self._lm = AutoModelForImageTextToText.from_pretrained(model, dtype="auto", device_map="auto")

    def generate(self, prompt: str, text: str = None, image_png: bytes = None) -> dict:
        import torch
        from PIL import Image

        content = []
        if image_png is not None:
            content.append({"type": "image", "image": Image.open(io.BytesIO(image_png)).convert("RGB")})
        content.append({"type": "text", "text": prompt if text is None else f"{prompt}\n\nReceipt OCR text:\n{text}"})
        inputs = self._proc.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        ).to(self._lm.device)
        with torch.inference_mode():
            out = self._lm.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # so the caller's timer includes all GPU work
        n_in = inputs["input_ids"].shape[1]
        new = out[:, n_in:]
        return {
            "text": self._proc.batch_decode(new, skip_special_tokens=True)[0],
            "input_tokens": int(n_in),
            "output_tokens": int(new.shape[1]),
        }
