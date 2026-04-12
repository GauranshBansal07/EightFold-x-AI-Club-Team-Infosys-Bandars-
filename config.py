"""
HireMind — Configuration & API Setup
Uses Groq API (free tier) with OpenAI-compatible SDK.
Supports dual API keys with automatic failover.
"""

import os
from openai import OpenAI

# ─── API Configuration ───────────────────────────────────────────────
# ⚠️ IMPORTANT: Replace these with your own Groq API keys.
#    Get free keys at https://console.groq.com/keys
#    Two keys are supported for automatic failover when rate limits hit.
GROQ_API_KEY_1 = "YOUR_GROQ_API_KEY_1"
GROQ_API_KEY_2 = "YOUR_GROQ_API_KEY_2"

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MODEL_NAME = "llama-3.3-70b-versatile"

# Active key tracker (module-level state for failover)
_active_key_index = 0

# ─── Model Parameters ────────────────────────────────────────────────
TEMPERATURE = 0.3          # Low temperature for consistent, structured output
MAX_TOKENS = 2048          # Sufficient for JSON responses
RETRY_ATTEMPTS = 2         # Retry once on failure per key
RETRY_DELAY_BASE = 2       # Exponential backoff base (seconds)

# ─── Interview Constants ─────────────────────────────────────────────
MAX_PROBES_PER_COMPETENCY = 4   # Including initial question (3 follow-ups max)
MAX_INTRODUCTION_EXCHANGES = 2  # Maximum exchanges in introduction phase
MIN_COMPETENCIES = 3
MAX_COMPETENCIES = 5

# ─── File Paths ───────────────────────────────────────────────────────
STATE_FILES_DIR = os.path.join(os.path.dirname(__file__), "state_files")
CANDIDATE_STATE_FILE = os.path.join(STATE_FILES_DIR, "candidate_state.md")
INTERVIEW_LOG_FILE = os.path.join(STATE_FILES_DIR, "interview_log.md")


def _get_api_keys() -> list[str]:
    """Return list of available API keys, filtering out empty strings."""
    keys = [GROQ_API_KEY_1, GROQ_API_KEY_2]
    return [k for k in keys if k and k.strip()]


def get_client(api_key: str = None) -> OpenAI:
    """Create and return the Groq API client (OpenAI-compatible)."""
    global _active_key_index
    if api_key is None:
        keys = _get_api_keys()
        api_key = keys[_active_key_index % len(keys)] if keys else ""
    return OpenAI(
        api_key=api_key,
        base_url=GROQ_BASE_URL,
    )


def _switch_key():
    """Switch to the other API key after a rate limit error."""
    global _active_key_index
    keys = _get_api_keys()
    if len(keys) > 1:
        _active_key_index = (_active_key_index + 1) % len(keys)
        return True
    return False


def llm_call(system_prompt: str, user_prompt: str, temperature: float = None) -> str:
    """
    Make a single LLM call with retry logic, dual-key failover, and JSON validation.
    Returns the raw text content from the model.

    On rate-limit (429), automatically switches to the backup API key.
    """
    import time

    keys = _get_api_keys()
    temp = temperature if temperature is not None else TEMPERATURE

    last_error = None

    # Try each key
    for key_attempt in range(len(keys)):
        client = get_client()

        for attempt in range(RETRY_ATTEMPTS):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temp,
                    max_tokens=MAX_TOKENS,
                )
                content = response.choices[0].message.content.strip()
                return content

            except Exception as e:
                last_error = e
                error_str = str(e)

                # On rate limit, try switching keys immediately
                if "429" in error_str or "rate_limit" in error_str.lower():
                    switched = _switch_key()
                    if switched:
                        client = get_client()  # Get client with new key
                        break  # Break inner retry loop to try with new key
                
                if attempt < RETRY_ATTEMPTS - 1:
                    wait = RETRY_DELAY_BASE ** (attempt + 1)
                    time.sleep(wait)
        else:
            # Inner loop exhausted without breaking — try switching key
            if not _switch_key():
                break  # No more keys to try
            continue

    raise RuntimeError(f"LLM call failed after trying all keys: {last_error}")


def llm_call_json(system_prompt: str, user_prompt: str, temperature: float = None) -> dict:
    """
    Make an LLM call expecting JSON output. Parses and validates the response.
    On parse failure, retries with a stricter prompt.
    """
    import json

    content = llm_call(system_prompt, user_prompt, temperature)

    # Try to extract JSON from the response (handles markdown code fences)
    json_str = content
    if "```json" in json_str:
        json_str = json_str.split("```json")[1].split("```")[0].strip()
    elif "```" in json_str:
        json_str = json_str.split("```")[1].split("```")[0].strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Retry with stricter instruction
        strict_system = system_prompt + "\n\nCRITICAL: You MUST respond with ONLY valid JSON. No explanations, no markdown, no code fences. Just the raw JSON object."
        content = llm_call(strict_system, user_prompt, temperature)

        json_str = content
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()

        return json.loads(json_str)
