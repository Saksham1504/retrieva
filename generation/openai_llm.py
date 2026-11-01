"""A tiny OpenAI wrapper: llm(instructions, user_input) -> text, with an on-disk cache.

The API key is read from the OPENAI_API_KEY environment variable (never put it in code).
The model comes from the `model` argument, else the OPENAI_MODEL environment variable,
else DEFAULT_MODEL. Identical calls are answered from the cache, so re-running an
evaluation does not cost anything again.
"""
import hashlib
import json
import os
from pathlib import Path

DEFAULT_MODEL = "gpt-5-mini"   # change with OPENAI_MODEL if your account uses a different model


class OpenAILLM:
    def __init__(self, model=None, cache_path=None, max_output_tokens=4000, client=None):
        self.model = model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
        self.max_output_tokens = max_output_tokens
        if client is None:
            from openai import OpenAI          # slow import, so only when needed
            client = OpenAI()                  # reads OPENAI_API_KEY
        self.client = client
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache = {}
        if self.cache_path and self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        self.calls = 0                         # real API calls made (cache hits are free)

    def _key(self, instructions, user_input):
        payload = f"{self.model}\n{instructions}\n{user_input}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def __call__(self, instructions, user_input):
        key = self._key(instructions, user_input)
        if key in self.cache:
            return self.cache[key]

        # Reasoning models spend part of max_output_tokens on hidden thinking, so keep it generous.
        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=user_input,
            max_output_tokens=self.max_output_tokens,
        )
        self.calls += 1
        text = (response.output_text or "").strip()
        if not text:
            raise RuntimeError(
                f"{self.model} returned no text (status: {getattr(response, 'status', '?')}). "
                "A reasoning model may have used all of max_output_tokens; try raising it.")

        self.cache[key] = text
        if self.cache_path:
            # Merge with what is on disk: the generator and the judge share one cache file.
            merged = {}
            if self.cache_path.exists():
                merged = json.loads(self.cache_path.read_text(encoding="utf-8"))
            merged.update(self.cache)
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(merged), encoding="utf-8")
            self.cache = merged
        return text
