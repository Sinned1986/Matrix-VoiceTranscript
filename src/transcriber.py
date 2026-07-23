from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class Transcriber:
    """Wraps the NeMo Parakeet TDT model for CPU ASR inference.

    Load once at startup via Transcriber.load(), then call
    transcribe_wav() for each audio file. This is a synchronous
    class - all methods are blocking and must be called via run_in_executor.
    """

    def __init__(self, model: object) -> None:
        self._model = model

    @classmethod
    def load(cls, model_name: str) -> Transcriber:
        import nemo.collections.asr as nemo_asr
        import torch

        logger.info("Loading ASR model: %s", model_name)
        model = nemo_asr.models.ASRModel.from_pretrained(model_name, map_location="cpu")
        model = model.to(torch.device("cpu"))
        model.eval()
        logger.info("ASR model loaded on CPU")
        return cls(model)

    def transcribe_wav(self, wav_path: str) -> str:
        """Transcribe a 16 kHz mono WAV file. Blocking - call via run_in_executor."""
        results = self._model.transcribe([wav_path])
        if not results:
            return ""
        result = results[0]
        if hasattr(result, "text"):
            return result.text
        return str(result)

    def transcribe_wav_chunks(self, wav_paths: list[str]) -> str:
        """Transcribe multiple 16 kHz mono WAV chunks and join the text.

        Feeding the whole file to the model in one call keeps the entire
        audio (and its intermediate activations) in memory at once, which
        OOM-kills long recordings. Chunks are transcribed in one batched
        model.transcribe() call, then joined in order.
        Blocking - call via run_in_executor.
        """
        results = self._model.transcribe(wav_paths)
        texts = []
        for result in results:
            text = result.text if hasattr(result, "text") else str(result)
            if text and text.strip():
                texts.append(text.strip())
        return " ".join(texts)
