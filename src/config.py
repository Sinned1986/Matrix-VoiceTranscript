from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

MATRIX_NIO_DEVICE_ID = "MATRIX_VOICE_TRANSCRIPT"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    matrix_hs_url: str
    matrix_user_id: str
    matrix_access_token: str
    matrix_password: str | None = None

    locale: str = "en"

    asr_model_name: str = "nvidia/parakeet-tdt-0.6b-v3"

    max_audio_bytes: int = 25 * 1024 * 1024

    # Long recordings are split into chunks before ASR inference - feeding
    # the whole file in one model.transcribe() call keeps all activations
    # for the entire duration in memory at once, which OOM-kills the
    # container on CPU for recordings much past a few minutes.
    asr_chunk_seconds: int = 60

    store_path: str = "/data/store"

    def matrix_homeserver_base(self) -> str:
        return self.matrix_hs_url.rstrip("/")
