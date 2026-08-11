"""Typed, validated application configuration.

Replaces the previous module-of-globals with a single immutable `Settings`
object. Nothing in the codebase reads `os.environ` directly, which means every
component can be constructed with an explicit configuration in tests instead of
mutating process-wide state.

Environment variable names match the field names (case-insensitive), so
`OCR_ENGINE=vl` sets `settings.ocr_engine`. Values can also come from a `.env`
file in the working directory.
"""

from __future__ import annotations

import functools
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AliasChoices, Field, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".pdf"}
)

#: Extension -> accepted leading magic bytes. Used to reject files whose
#: contents do not match their claimed type before any decoder touches them.
MAGIC_PREFIXES: dict[str, tuple[bytes, ...]] = {
    ".pdf": (b"%PDF-",),
    ".png": (b"\x89PNG\r\n\x1a\n",),
    ".jpg": (b"\xff\xd8\xff",),
    ".jpeg": (b"\xff\xd8\xff",),
    ".bmp": (b"BM",),
    ".tif": (b"II*\x00", b"MM\x00*"),
    ".tiff": (b"II*\x00", b"MM\x00*"),
    ".webp": (b"RIFF",),
}


class Environment(StrEnum):
    development = "development"
    staging = "staging"
    production = "production"


class ExtractionMode(StrEnum):
    """How invoice rows are derived from OCR output.

    hybrid
        Deterministic parser first; the LLM is consulted only when the parser
        finds nothing or its arithmetic checksum fails.
    deterministic
        Never call the LLM. Fully reproducible and the safest default for
        regulated review workflows.
    llm
        LLM only. Diagnostic use; small models routinely merge invoice rows.
    """

    hybrid = "hybrid"
    deterministic = "deterministic"
    llm = "llm"


class OCREngineName(StrEnum):
    paddleocr = "paddleocr"
    vl = "vl"


Port = Annotated[int, Field(ge=1, le=65535)]


class Settings(BaseSettings):
    """Immutable application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
        # Aliased fields (API_KEYS -> api_keys_raw) stay constructible by field
        # name, which keeps test setup readable.
        populate_by_name=True,
    )

    # ----------------------------------------------------------------- app --
    environment: Environment = Environment.production
    debug: bool = False

    # --------------------------------------------------------------- paths --
    data_dir: Path = Path("data")
    invoices_dir_override: Path | None = Field(default=None, validation_alias="INVOICES_DIR")
    outputs_dir_override: Path | None = Field(default=None, validation_alias="OUTPUTS_DIR")
    logs_dir_override: Path | None = Field(default=None, validation_alias=AliasChoices("LOG_DIR", "LOGS_DIR"))
    temp_dir_override: Path | None = Field(
        default=None, validation_alias=AliasChoices("OCR_TEMP_DIR", "TEMP_DIR")
    )
    cache_dir_override: Path | None = Field(default=None, validation_alias="OCR_CACHE_DIR")

    # ----------------------------------------------------------------- http --
    host: str = "127.0.0.1"
    port: Port = 8000
    #: Comma-separated API keys. Empty disables authentication entirely.
    api_keys_raw: str = Field(default="", validation_alias="API_KEYS")
    #: Set when authentication is terminated by a trusted upstream gateway.
    allow_unauthenticated: bool = False

    # ----------------------------------------------------------------- tls --
    #: Serve HTTPS directly. Usually TLS is better terminated at a reverse
    #: proxy, but a standalone deployment needs this.
    ssl_certfile: Path | None = None
    ssl_keyfile: Path | None = None
    cors_origins_raw: str = Field(default="", validation_alias="CORS_ORIGINS")
    docs_enabled: bool = False
    max_upload_bytes: int = Field(default=20 * 1024 * 1024, ge=1024)
    #: Ceiling on one folder upload. Files are still individually capped by
    #: `max_upload_bytes`; these two bound the request as a whole.
    max_batch_files: int = Field(default=50, ge=1, le=1000)
    max_batch_upload_bytes: int = Field(default=200 * 1024 * 1024, ge=1024)
    rate_limit_per_minute: int = Field(default=60, ge=0)
    max_concurrent_extractions: int = Field(default=2, ge=1, le=64)
    request_timeout_s: float = Field(default=300.0, gt=0)
    #: Trust X-Forwarded-For for client identity. Only enable behind a proxy
    #: that overwrites the header, otherwise rate limits are trivially bypassed.
    trust_proxy_headers: bool = False

    # ------------------------------------------------------------ pipeline --
    extraction_mode: ExtractionMode = ExtractionMode.hybrid
    photo_mode: bool = False
    apply_adaptive_threshold: bool = True
    batch_concurrency: int = Field(default=2, ge=1, le=32)

    # ----------------------------------------------------------------- ocr --
    ocr_engine: OCREngineName = OCREngineName.paddleocr
    #: "auto" uses the GPU when PaddlePaddle was built with CUDA and a device is
    #: present, else CPU. Set "cpu" or "gpu" to force it.
    ocr_device: str = "auto"
    ocr_lang: str = "fr"
    ocr_version: str = "PP-OCRv5"
    ocr_text_detection_model_name: str | None = "PP-OCRv5_mobile_det"
    ocr_text_recognition_model_name: str | None = "latin_PP-OCRv5_mobile_rec"
    ocr_text_recognition_batch_size: int = Field(default=8, ge=1, le=256)
    ocr_min_confidence: float = Field(default=0.35, ge=0.0, le=1.0)
    paddle_disable_mkldnn: bool = True
    warmup_on_startup: bool = True
    #: Re-run OCR upright when the first pass returns sideways text boxes.
    #: PaddleOCR's own classifier misses some scans; costs one extra pass on
    #: the few pages that need it.
    correct_page_orientation: bool = True
    #: Extra OCR passes the recovery sweep may spend on a page that parsed
    #: nothing. Each pass is a full re-OCR (60-80s in photo mode), so an
    #: unbounded sweep over three angles put a single invoice at ~5 minutes.
    #: 180 is tried first because upside-down is by far the common case.
    max_orientation_probes: int = Field(default=1, ge=0, le=3)
    use_doc_orientation_classify: bool = True
    use_doc_unwarping: bool = False
    use_textline_orientation: bool = True

    # --------------------------------------------------------------- cache --
    use_ocr_cache: bool = True
    force_ocr_refresh: bool = False
    ocr_cache_version: str = "v3"
    ocr_cache_max_entries: int = Field(default=5000, ge=0)
    ocr_cache_ttl_days: int = Field(default=30, ge=0)

    # ------------------------------------------------------- preprocessing --
    target_long_side: int = Field(default=2600, ge=500, le=10000)
    min_long_side: int = Field(default=1500, ge=100, le=10000)
    max_long_side: int = Field(default=3300, ge=500, le=12000)
    clahe_clip_limit: float = Field(default=2.0, gt=0)
    clahe_tile_grid: int = Field(default=8, ge=1, le=64)
    denoise_strength: int = Field(default=7, ge=1, le=50)
    adaptive_block_size: int = Field(default=31, ge=3, le=255)
    adaptive_c: int = 15
    perspective_min_area_ratio: float = Field(default=0.25, gt=0, le=1.0)
    #: Refuse images larger than this many megapixels (decompression-bomb guard).
    max_image_megapixels: float = Field(default=80.0, gt=0)

    # ----------------------------------------------------------------- pdf --
    pdf_render_dpi: int = Field(default=220, ge=72, le=600)
    pdf_max_pages: int = Field(default=5, ge=1, le=100)
    retry_ocr_on_review_flags: bool = True
    pdf_retry_render_dpi: int = Field(default=260, ge=72, le=600)
    pdf_retry_target_long_side: int = Field(default=3000, ge=500, le=10000)
    pdf_retry_max_long_side: int = Field(default=3600, ge=500, le=12000)
    enable_deep_ocr_retry: bool = True
    pdf_deep_retry_render_dpi: int = Field(default=300, ge=72, le=600)
    pdf_deep_retry_target_long_side: int = Field(default=3300, ge=500, le=10000)
    pdf_deep_retry_max_long_side: int = Field(default=4200, ge=500, le=12000)

    # -------------------------------------------------------------- ollama --
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:3b"
    ollama_num_predict: int = Field(default=4096, ge=64)
    ollama_num_ctx: int = Field(default=8192, ge=256)
    ollama_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    ollama_timeout_s: float = Field(default=180.0, gt=0)
    ollama_max_retries: int = Field(default=2, ge=0, le=10)

    # ---------------------------------------------------------- validation --
    refcode_min_length: int = Field(default=4, ge=1)
    #: Absolute dinar tolerance when comparing summed line totals to the
    #: printed invoice total. Tunisian invoices are quoted in millimes.
    checksum_tolerance: float = Field(default=0.05, ge=0)

    # ---------------------------------------------------------------- jobs --
    job_retention_minutes: int = Field(default=60, ge=1)
    job_queue_max_size: int = Field(default=100, ge=1)

    # ------------------------------------------------------------- logging --
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["json", "text"] = "json"
    log_to_file: bool = True
    log_max_bytes: int = Field(default=5 * 1024 * 1024, ge=1024)
    log_backup_count: int = Field(default=3, ge=0)

    # ----------------------------------------------------------------------- #
    # Derived values
    # ----------------------------------------------------------------------- #
    @computed_field  # type: ignore[prop-decorator]
    @property
    def invoices_dir(self) -> Path:
        return self.invoices_dir_override or self.data_dir / "invoices"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def outputs_dir(self) -> Path:
        return self.outputs_dir_override or self.data_dir / "outputs"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def logs_dir(self) -> Path:
        return self.logs_dir_override or self.data_dir / "logs"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def temp_dir(self) -> Path:
        return self.temp_dir_override or self.data_dir / "tmp"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ocr_cache_dir(self) -> Path:
        return self.cache_dir_override or self.data_dir / "ocr_cache"

    @property
    def tls_enabled(self) -> bool:
        return self.ssl_certfile is not None and self.ssl_keyfile is not None

    @property
    def api_keys(self) -> frozenset[str]:
        return frozenset(part.strip() for part in self.api_keys_raw.split(",") if part.strip())

    @property
    def cors_origins(self) -> tuple[str, ...]:
        return tuple(part.strip() for part in self.cors_origins_raw.split(",") if part.strip())

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_keys)

    @property
    def is_production(self) -> bool:
        return self.environment is Environment.production

    # ----------------------------------------------------------------------- #
    # Cross-field validation
    # ----------------------------------------------------------------------- #
    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        if self.min_long_side > self.max_long_side:
            raise ValueError("min_long_side must not exceed max_long_side")
        if self.adaptive_block_size % 2 == 0:
            raise ValueError("adaptive_block_size must be odd (OpenCV requirement)")

        if self.ssl_certfile is not None and self.ssl_keyfile is None:
            raise ValueError("SSL_CERTFILE requires SSL_KEYFILE")
        if self.ssl_keyfile is not None and self.ssl_certfile is None:
            raise ValueError("SSL_KEYFILE requires SSL_CERTFILE")
        if self.is_production:
            # Shipping an unauthenticated extraction endpoint that accepts
            # arbitrary uploads is the single most likely way this service gets
            # abused, so refuse to start rather than warn.
            if not self.auth_enabled and not self.allow_unauthenticated:
                raise ValueError(
                    "ENVIRONMENT=production requires API_KEYS to be set. "
                    "If authentication is handled by an upstream gateway, set "
                    "ALLOW_UNAUTHENTICATED=true to acknowledge this explicitly."
                )
            if self.debug:
                raise ValueError("DEBUG must be false in production")
            if "*" in self.cors_origins:
                raise ValueError("CORS_ORIGINS='*' is not allowed in production")
        return self

    def ensure_directories(self) -> None:
        """Create the runtime directories this process writes to."""
        for directory in (self.temp_dir, self.ocr_cache_dir, self.logs_dir, self.outputs_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def public_snapshot(self) -> dict[str, object]:
        """Non-secret configuration, safe to expose on the health endpoint."""
        return {
            "environment": self.environment.value,
            "extraction_mode": self.extraction_mode.value,
            "ocr_engine": self.ocr_engine.value,
            "ocr_device": self.ocr_device,
            "ocr_lang": self.ocr_lang,
            "ocr_version": self.ocr_version,
            "photo_mode": self.photo_mode,
            "use_ocr_cache": self.use_ocr_cache,
            "pdf_render_dpi": self.pdf_render_dpi,
            "pdf_max_pages": self.pdf_max_pages,
            "max_upload_bytes": self.max_upload_bytes,
            "max_batch_files": self.max_batch_files,
            "max_concurrent_extractions": self.max_concurrent_extractions,
            "auth_enabled": self.auth_enabled,
            "ollama_model": self.ollama_model,
        }


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton.

    Cached so importing modules never re-parse the environment. Tests should
    construct `Settings(...)` directly rather than clearing this cache.
    """
    return Settings()
