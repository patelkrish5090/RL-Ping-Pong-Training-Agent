"""
llm_advisor.py — LLM Importance Weight Advisor using Ollama (Qwen3-Coder).

Queries a local/remote Ollama server every K episodes (or on reward plateau)
to obtain per-feature importance weights for the CartPole state features:
    [x_cart, x_dot, theta, theta_dot]

These weights are used to compute the LLM-guided auxiliary reconstruction loss
in Variant A of the RL-over-Noisy-Channels system.
"""
from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import requests


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """\
You are a physics-aware feature importance advisor for a reinforcement learning system.

TASK: Balance a pole on a cart (CartPole). Episode ends if:
- |pole_angle| > 0.2095 radians (~12 degrees)
- |cart_position| > 2.4 units
- Steps > 200

STATE FEATURES (index 0-3):
0: cart_position (range ±4.8, terminal at ±2.4)
1: cart_velocity (unbounded)
2: pole_angle (range ±0.418 rad, terminal at ±0.2095)
3: pole_angular_velocity (unbounded)

CURRENT CHANNEL CONDITIONS:
- Bottleneck dimension n: {n}
- Channel noise variance σ²: {sigma2}
- Channel capacity upper bound: n * log2(1 + 1/σ²) bits = {capacity:.2f} bits

RECENT TRAINING PERFORMANCE:
- Last 20 episodes mean reward: {recent_mean:.1f} / 200
- Last 100 episodes mean reward: {longer_mean:.1f} / 200
- Training episode: {episode}

Given these constraints, assign importance weights to each state feature for compression.
Features with higher weight should be preserved more accurately through the noisy channel.
Weights must sum to 1.0 and be between 0.05 and 0.60.

Physics reasoning: Under tight bandwidth (small n) and high noise (large σ²),
the encoder must prioritize features that most directly determine episode termination.

Respond ONLY with valid JSON, no explanation, no markdown:
{{"weights": [w0, w1, w2, w3], "reasoning": "one sentence max"}}"""


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LLMAdvisor:
    """
    Queries an Ollama-hosted LLM for per-feature importance weights.

    Parameters
    ----------
    variant_cfg : VariantAConfig
        Contains LLM model name, server URL, timeouts, query interval, etc.
    channel_cfg : ChannelConfig
        Used for building the capacity / noise info in the prompt.
    log_dir : str
        Directory where llm_log.jsonl will be written.
    """

    def __init__(self, variant_cfg, channel_cfg, log_dir: str) -> None:
        self._vcfg = variant_cfg
        self._ccfg = channel_cfg
        self._log_path = os.path.join(log_dir, "llm_log.jsonl")
        os.makedirs(log_dir, exist_ok=True)

        # State
        self._cached_weights: np.ndarray = self._fallback_weights()
        self._last_query_episode: int = -1
        self._last_query_mean: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_weights(
        self,
        episode: int,
        recent_rewards: List[float],
    ) -> np.ndarray:
        """
        Return the current LLM importance weights (shape: [4,]).

        Triggers a new query if:
        - K episodes have elapsed since last query, OR
        - A plateau is detected (20-ep moving avg dropped > threshold vs last query avg)

        Parameters
        ----------
        episode : int
            Current training episode index.
        recent_rewards : List[float]
            Full list of episode rewards so far (used for moving averages).

        Returns
        -------
        weights : np.ndarray, shape (4,), dtype float32, sums to 1.0
        """
        should_query = False

        # Interval-based trigger
        if (episode - self._last_query_episode) >= self._vcfg.llm_query_interval:
            should_query = True

        # Plateau-based early trigger
        if len(recent_rewards) >= self._vcfg.plateau_window:
            window = recent_rewards[-self._vcfg.plateau_window:]
            current_mean = float(np.mean(window))
            if (
                self._last_query_episode >= 0
                and (self._last_query_mean - current_mean) > self._vcfg.plateau_threshold
            ):
                should_query = True

        if should_query:
            weights = self._query_llm(episode, recent_rewards)
            self._cached_weights = weights
            self._last_query_episode = episode
            if len(recent_rewards) >= self._vcfg.plateau_window:
                self._last_query_mean = float(
                    np.mean(recent_rewards[-self._vcfg.plateau_window:])
                )

        return self._cached_weights

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _query_llm(self, episode: int, recent_rewards: List[float]) -> np.ndarray:
        """
        Send prompt to Ollama and return parsed + normalized weights.
        Falls back to physics-based defaults on any failure.
        """
        prompt = self._build_prompt(episode, recent_rewards)
        start_time = time.monotonic()
        source = "llm"
        reasoning = ""
        weights = None
        raw_response = ""

        for attempt in range(3):
            try:
                raw_response = self._call_ollama(prompt)
                weights = self._parse_response(raw_response)
                if weights is not None:
                    break
                else:
                    source = "fallback_parse_error"
                    print(f"[LLMAdvisor] Ep {episode}: Empty or invalid JSON. Retrying (attempt {attempt+1}/3)...")
            except requests.exceptions.Timeout:
                source = "fallback_timeout"
                print(f"[LLMAdvisor] Ep {episode}: Ollama request timed out — using fallback weights.")
                break
            except requests.exceptions.ConnectionError:
                source = "fallback_connection_error"
                print(f"[LLMAdvisor] Ep {episode}: Cannot reach Ollama server ({self._vcfg.ollama_url}) — using fallback weights.")
                break
            except Exception as exc:
                source = f"fallback_error:{type(exc).__name__}"
                print(f"[LLMAdvisor] Ep {episode}: Unexpected error ({exc}) — using fallback weights.")
                break

        if weights is None:
            weights = self._fallback_weights()
        else:
            # Try to extract reasoning from successful parse
            try:
                start_idx = raw_response.find("{")
                end_idx = raw_response.rfind("}")
                if start_idx != -1 and end_idx != -1:
                    data = json.loads(raw_response[start_idx:end_idx+1])
                    reasoning = str(data.get("reasoning", ""))[:200]
            except Exception:
                pass

        elapsed = time.monotonic() - start_time
        self._log(episode, weights, reasoning, source, elapsed, raw_response)
        return weights

    def _build_prompt(self, episode: int, recent_rewards: List[float]) -> str:
        n = self._ccfg.n
        sigma2 = self._ccfg.sigma2_channel

        # Channel capacity (Shannon formula, cap at 99.9 for zero noise)
        if sigma2 <= 0.0:
            capacity = 99.9
        else:
            capacity = n * math.log2(1.0 + 1.0 / sigma2)

        recent_mean = float(np.mean(recent_rewards[-20:])) if len(recent_rewards) >= 20 else (
            float(np.mean(recent_rewards)) if recent_rewards else 0.0
        )
        longer_mean = float(np.mean(recent_rewards[-100:])) if len(recent_rewards) >= 100 else (
            float(np.mean(recent_rewards)) if recent_rewards else 0.0
        )

        return _PROMPT_TEMPLATE.format(
            n=n,
            sigma2=sigma2,
            capacity=capacity,
            recent_mean=recent_mean,
            longer_mean=longer_mean,
            episode=episode,
        )

    def _call_ollama(self, prompt: str) -> str:
        """POST to Ollama /api/chat endpoint and return the model's text reply."""
        url = self._vcfg.ollama_url.rstrip("/") + "/api/chat"
        payload = {
            "model": self._vcfg.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "num_predict": 2048,   # High limit because DeepSeek-R1 outputs long <think> blocks
                "temperature": 0.2,    # Low temperature for deterministic JSON output
            },
        }
        response = requests.post(
            url,
            json=payload,
            timeout=self._vcfg.api_timeout,
        )
        response.raise_for_status()
        data = response.json()
        # Ollama /api/chat response: {"message": {"role": "assistant", "content": "..."}}
        return data["message"]["content"]

    def _parse_response(self, text: str) -> Optional[np.ndarray]:
        """
        Parse LLM JSON response → normalized weight array, or None on failure.

        Handles accidental markdown code fences around the JSON.
        """
        try:
            cleaned = text.strip()
            
            # Strip DeepSeek <think>...</think> reasoning blocks
            if "</think>" in cleaned:
                cleaned = cleaned.split("</think>")[-1].strip()

            # Bulletproof extraction: find the first { and last }
            start_idx = cleaned.find("{")
            end_idx = cleaned.rfind("}")
            
            if start_idx != -1 and end_idx != -1:
                cleaned = cleaned[start_idx:end_idx+1]
            else:
                raise ValueError("No JSON object found in response")

            data = json.loads(cleaned)
            weights = np.array(data["weights"], dtype=np.float32)
            assert len(weights) == 4, f"Expected 4 weights, got {len(weights)}"
            assert all(0.0 <= float(w) <= 1.0 for w in weights), "Weights out of [0,1] range"

            # Clamp and normalize
            weights = np.clip(weights, 0.05, 0.60)
            weights = weights / weights.sum()
            return weights

        except Exception as exc:
            print(f"[LLMAdvisor] Parse error: {exc!r} | raw: {text[:200]!r}")
            return None

    def _fallback_weights(self) -> np.ndarray:
        """Physics-based default weights when LLM is unavailable."""
        sigma2 = self._ccfg.sigma2_channel
        if sigma2 < self._vcfg.noise_threshold:
            w = np.array(self._vcfg.fallback_weights_low_noise, dtype=np.float32)
        else:
            w = np.array(self._vcfg.fallback_weights_high_noise, dtype=np.float32)
        # Normalize (should already sum to 1, but enforce defensively)
        w = np.clip(w, 0.05, 0.60)
        return w / w.sum()

    def _log(
        self,
        episode: int,
        weights: np.ndarray,
        reasoning: str,
        source: str,
        elapsed: float,
        raw_response: str,
    ) -> None:
        """Append one entry to llm_log.jsonl."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "episode": episode,
            "weights": weights.tolist(),
            "reasoning": reasoning,
            "source": source,
            "elapsed_sec": round(elapsed, 3),
            "raw_response_snippet": raw_response[:300],
        }
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
