"""Resilient Vertex AI Express REST client for Gemini 3.8 Flash.

Features:
- Direct HTTP REST communication via Vertex AI Express endpoints.
- Full-jitter exponential backoff retry mechanism for HTTP 429 / RESOURCE_EXHAUSTED / 5xx.
- Configurable thinking budget (0 for rapid scoring, 1024 for deep reasoning).
- Preprocessed base64 image encoding with aspect-ratio-aware databar removal.
- Robust JSON response extraction from model outputs.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from pathlib import Path
import random
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import httpx
from PIL import Image

from assume.config import VERTEX_API_KEY, VERTEX_MODEL, VERTEX_PROJECT_ID

logger = logging.getLogger(__name__)

# Default model configuration
DEFAULT_MODEL: str = VERTEX_MODEL
DEFAULT_BASE_DELAY: float = 1.0
DEFAULT_MAX_DELAY: float = 60.0
DEFAULT_MAX_RETRIES: int = 7
DEFAULT_TIMEOUT: float = 60.0
RETRYABLE_STATUS_CODES: Tuple[int, ...] = (429, 500, 502, 503, 504)


class ResilientVertexClient:
    """Robust REST client communicating directly with Gemini on Vertex AI Express."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_delay: float = DEFAULT_BASE_DELAY,
        max_delay: float = DEFAULT_MAX_DELAY,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """Initialize the resilient Vertex AI Express client.

        Args:
            api_key: Vertex API key. If omitted, reads VERTEX_API_KEY from env.
            project_id: Vertex project ID. If omitted, reads VERTEX_PROJECT_ID from env.
            model: Gemini model identifier (default: gemini-3.8-flash or VERTEX_MODEL).
            max_retries: Maximum backoff retry attempts for rate limits / 5xx.
            base_delay: Base exponential backoff delay in seconds.
            max_delay: Maximum ceiling for exponential backoff delay.
            timeout: HTTP request timeout in seconds.
        """
        self.api_key: Optional[str] = api_key or os.environ.get("VERTEX_API_KEY")
        self.project_id: Optional[str] = project_id or os.environ.get("VERTEX_PROJECT_ID")
        raw_model = model or os.environ.get("VERTEX_MODEL", DEFAULT_MODEL)
        if "publishers/google/models/" in raw_model:
            raw_model = raw_model.split("publishers/google/models/")[-1]
        self.model: str = raw_model.strip().strip("/")

        self.max_retries: int = max_retries
        self.base_delay: float = base_delay
        self.max_delay: float = max_delay
        self.timeout: float = timeout

        if not self.api_key or not self.project_id:
            raise ValueError(
                "Both VERTEX_API_KEY and VERTEX_PROJECT_ID must be provided "
                "or present in environment variables."
            )

        self.endpoint: str = (
            f"https://aiplatform.googleapis.com/v1/projects/{self.project_id}"
            f"/locations/global/publishers/google/models/{self.model}:generateContent"
        )
        self.headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

    def compute_backoff(self, attempt: int, retry_after_header: Optional[str] = None) -> float:
        """Compute sleep duration using full-jitter exponential backoff.

        Formula:
            ceiling = min(max_delay, base_delay * 2^(attempt - 1))
            sleep = uniform(0.05, ceiling)
        If 'Retry-After' header is supplied, honors retry_after + jitter.

        Args:
            attempt: 1-based attempt index.
            retry_after_header: Optional string from response HTTP header.

        Returns:
            Calculated sleep duration in seconds.
        """
        if retry_after_header:
            try:
                seconds = float(retry_after_header)
                return max(0.0, seconds) + random.uniform(0.1, 1.0)
            except ValueError:
                pass

        ceiling = min(self.max_delay, self.base_delay * (2.0 ** (attempt - 1)))
        low = min(0.05, ceiling)
        return random.uniform(low, ceiling)

    def call(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a generateContent call with retry and full-jitter exponential backoff.

        Args:
            payload: JSON request payload conforming to Gemini generateContent schema.

        Returns:
            Decoded JSON response dictionary from Vertex AI.

        Raises:
            RuntimeError: If retries are exhausted or a non-retryable error occurs.
        """
        attempt = 0
        with httpx.Client(timeout=self.timeout) as client:
            while True:
                try:
                    resp = client.post(self.endpoint, headers=self.headers, json=payload)
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        # Verify whether the body itself encapsulates an error payload
                        if "error" in data:
                            err_info = data["error"]
                            code = err_info.get("code")
                            status = err_info.get("status", "")
                            if code in RETRYABLE_STATUS_CODES or "RESOURCE_EXHAUSTED" in status:
                                attempt += 1
                                if attempt > self.max_retries:
                                    raise RuntimeError(
                                        f"Max retries ({self.max_retries}) exceeded for error: {err_info}"
                                    )
                                sleep_time = self.compute_backoff(attempt)
                                logger.warning(
                                    "Rate limit in response body (%s). Retrying in %.2fs (attempt %d/%d)...",
                                    status, sleep_time, attempt, self.max_retries
                                )
                                time.sleep(sleep_time)
                                continue
                            raise RuntimeError(f"Vertex AI returned error in body: {err_info}")
                        return data

                    # Handle retryable HTTP codes
                    if resp.status_code in RETRYABLE_STATUS_CODES:
                        attempt += 1
                        if attempt > self.max_retries:
                            raise RuntimeError(
                                f"Max retries ({self.max_retries}) exceeded for HTTP {resp.status_code}: {resp.text}"
                            )
                        retry_after = resp.headers.get("retry-after")
                        sleep_time = self.compute_backoff(attempt, retry_after)
                        err_snippet = resp.text[:250].replace("\n", " ") if resp.text else ""
                        logger.warning(
                            "HTTP %d from Vertex AI [%s]. Retrying in %.2fs (attempt %d/%d)...",
                            resp.status_code, err_snippet, sleep_time, attempt, self.max_retries
                        )
                        time.sleep(sleep_time)
                        continue

                    # Non-retryable HTTP error (e.g., 400 Invalid Argument, 401/403 Auth)
                    raise RuntimeError(
                        f"Vertex AI API returned non-retryable error {resp.status_code}: {resp.text}"
                    )

                except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as net_err:
                    attempt += 1
                    if attempt > self.max_retries:
                        raise RuntimeError(
                            f"Max retries ({self.max_retries}) exceeded on network error: {net_err}"
                        ) from net_err
                    sleep_time = self.compute_backoff(attempt)
                    logger.warning(
                        "Network error (%s). Retrying in %.2fs (attempt %d/%d)...",
                        net_err, sleep_time, attempt, self.max_retries
                    )
                    time.sleep(sleep_time)

    def generate_content(
        self,
        contents: Sequence[Dict[str, Any]],
        system_instruction: Optional[str] = None,
        thinking_budget: Optional[int] = None,
        max_output_tokens: int = 4096,
        temperature: float = 0.2,
        response_mime_type: str = "application/json",
    ) -> str:
        """High-level generateContent invocation returning candidate text.

        Args:
            contents: List of content role/parts dictionaries.
            system_instruction: Optional system instruction prompt.
            thinking_budget: Optional integer thinking budget. 0 disables thinking.
            max_output_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
            response_mime_type: Desired MIME type (default application/json).

        Returns:
            Generated text string from candidate output.
        """
        payload: Dict[str, Any] = {
            "contents": list(contents),
            "generationConfig": {
                "responseMimeType": response_mime_type,
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
            ],
        }

        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        if thinking_budget is not None:
            payload["generationConfig"]["thinkingConfig"] = {
                "thinkingBudget": thinking_budget
            }

        raw = self.call(payload)
        return self.extract_text_from_response(raw)

    @staticmethod
    def extract_text_from_response(raw_resp: Dict[str, Any]) -> str:
        """Extract generated model output text, filtering out thinking thoughts if present."""
        candidates = raw_resp.get("candidates", [])
        if not candidates:
            raise ValueError(f"No candidates present in response: {raw_resp}")

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        if not parts:
            raise ValueError(f"No content parts found in candidate: {candidates[0]}")

        # In Gemini with internal reasoning, parts may have 'thought': True. Filter them.
        output_parts = [
            p.get("text", "")
            for p in parts
            if not p.get("thought", False) and "text" in p
        ]
        if output_parts:
            return "".join(output_parts).strip()

        # Fallback to all text parts if all were labeled thoughts
        return "".join(p.get("text", "") for p in parts if "text" in p).strip()

    @staticmethod
    def extract_clean_json(text: str) -> Any:
        """Extract JSON structure from model text, stripping markdown blocks if present.

        Handles:
            ```json [ ... ] ```
            ``` { ... } ```
            Raw JSON string with surrounding conversational text.

        Args:
            text: Raw string containing JSON.

        Returns:
            Parsed Python object (dict or list).
        """
        text = text.strip()
        # 1. Check for fenced code blocks
        fenced_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if fenced_match:
            candidate = fenced_match.group(1).strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # 2. Direct JSON attempt
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 3. Search for outermost brackets: {...} or [...]
        start_brace = text.find("{")
        start_bracket = text.find("[")

        if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
            end_brace = text.rfind("}")
            if end_brace > start_brace:
                return json.loads(text[start_brace : end_brace + 1])
        elif start_bracket != -1:
            end_bracket = text.rfind("]")
            if end_bracket > start_bracket:
                return json.loads(text[start_bracket : end_bracket + 1])

        # If all extraction fails, raise with clear context
        raise ValueError(f"Failed to extract valid JSON from response text: {text[:300]}...")

    @staticmethod
    def encode_image(
        image_or_path: Union[Image.Image, str, Path],
        target_size: Tuple[int, int] = (512, 288),
        quality: int = 85,
    ) -> str:
        """Load and preprocess a SEM micrograph to base64 JPEG string.

        Applies databar excision (top 75% height) and Lanczos resize to 512x288,
        guaranteeing exactly 1,100 vision tokens per image in Gemini 3.8 Flash.

        Args:
            image_or_path: PIL Image object or file path.
            target_size: Desired output dimensions (width, height).
            quality: JPEG compression quality factor.

        Returns:
            Base64 encoded ASCII string of the JPEG image.
        """
        try:
            from assume.data.ingestion import preprocess_micrograph
            img = preprocess_micrograph(image_or_path, target_size=target_size)
        except Exception:
            # Fallback if ingestion module is not loaded or during isolated testing
            if isinstance(image_or_path, (str, Path)):
                with Image.open(image_or_path) as opened_img:
                    img = opened_img.copy()
            else:
                img = image_or_path

            already_preprocessed = (
                getattr(img, "_is_preprocessed", False)
                or bool(getattr(img, "info", {}).get("preprocessed", False))
                or img.size == target_size
            )

            if img.mode != "RGB":
                img = img.convert("RGB")

            if not already_preprocessed:
                w, h = img.size
                crop_height = max(1, int(round(h * 0.75)))
                cropped = img.crop((0, 0, w, crop_height))
                img = cropped.resize(target_size, Image.Resampling.BILINEAR)
            elif img.size != target_size:
                img = img.resize(target_size, Image.Resampling.BILINEAR)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
