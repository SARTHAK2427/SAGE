import time
import httpx
from typing import List, Dict, Any, Optional
import config

class ModelClient:
    def __init__(self, base_url: str = config.LLAMA_BASE_URL, timeout: float = config.REQUEST_TIMEOUT):
        self.base_url = base_url
        self.timeout = timeout

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        extra_body: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Sends a standard or multimodal chat completion request to llama-server.
        Returns a dict containing:
        - 'content': generated text string
        - 'duration': total round-trip time in seconds
        - 'usage': token usage dict (prompt_tokens, completion_tokens, total_tokens)
        - 'timings': llama.cpp performance timings if available
        - 'raw': complete raw response dict
        """
        url = f"{self.base_url}/v1/chat/completions"
        payload: Dict[str, Any] = {
            "messages": messages,
            "stream": False
        }

        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if extra_body:
            payload.update(extra_body)

        start_time = time.time()
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"llama-server HTTP error {e.response.status_code}: {e.response.text}") from e
        except httpx.RequestError as e:
            raise RuntimeError(f"Failed to connect to llama-server at {url}: {e}") from e

        duration = time.time() - start_time

        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "")
        usage = data.get("usage", {})
        timings = data.get("timings", {})

        return {
            "content": content,
            "duration": duration,
            "usage": usage,
            "timings": timings,
            "raw": data
        }

# Global client instance
model_client = ModelClient()
