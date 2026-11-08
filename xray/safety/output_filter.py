from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from xray.evidence import EvidenceRegistry


@dataclass
class OutputFilterResult:
    is_clean: bool
    violations: List[str] = field(default_factory=list)


class OutputSafetyFilter:
    """
    Code-level safety guardrail enforcing non-predictive and non-advisory invariants.
    Blocks:
      1. Direct buy/sell execution instructions or trading recommendations.
      2. Forward-looking stock price predictions, price targets, or guaranteed return claims.
      3. Numerical hallucinations (numbers in agent explanations not grounded in Evidence).
    """

    PROHIBITED_TRADING_PATTERNS: List[Tuple[str, str]] = [
        (r"\byou\s+(?:should|must|need\s+to)\s+(?:buy|sell|short|liquidate)\b", "Direct buy/sell instruction"),
        (r"\b(?:recommend|advise)\s+(?:buying|selling|shorting|liquidating)\b", "Advisory buy/sell recommendation"),
        (r"\b(?:strong\s+buy|strong\s+sell|buy\s+immediately|sell\s+immediately|dump\s+all)\b", "Urgent trading signal"),
        (r"\b(?:execute|place)\s+(?:a\s+)?(?:buy|sell|market|limit)\s+order\b", "Order execution instruction"),
    ]

    PROHIBITED_PREDICTION_PATTERNS: List[Tuple[str, str]] = [
        (r"\bprice\s+target\s+of\b", "Stock price target prediction"),
        (r"\bwill\s+(?:rally|surge|crash|climb|drop|reach|hit)\s+to\s+[\$£€]?\d+", "Future price level prediction"),
        (r"\b(?:guaranteed|risk[-\s]free)\s+return\b", "Guaranteed return claim"),
        (r"\bpredicted\s+to\s+(?:rise|fall|outperform|double|triple)\b", "Forward return prediction"),
    ]

    def check_text(self, text: str) -> OutputFilterResult:
        """
        Scans text for prohibited trading instructions or price predictions.
        """
        violations: List[str] = []
        for pattern, label in self.PROHIBITED_TRADING_PATTERNS:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                violations.append(f"{label}: '{match.group(0)}'")

        for pattern, label in self.PROHIBITED_PREDICTION_PATTERNS:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                violations.append(f"{label}: '{match.group(0)}'")

        return OutputFilterResult(
            is_clean=len(violations) == 0,
            violations=violations,
        )

    @staticmethod
    def _collect_allowed_numbers(registry: EvidenceRegistry) -> List[float]:
        """
        Extracts all numeric values (and their percentage equivalents, e.g. 0.3471 -> 34.71)
        from registered Evidence objects and their inputs.
        """
        allowed: Set[float] = {0.0, 1.0, 3.0, 5.0, 10.0, 20.0, 60.0, 40.0, 95.0, 99.0, 100.0, 252.0, 756.0, 2020.0, 2022.0, 2023.0, 2024.0}

        def add_num(val: float) -> None:
            allowed.add(val)
            allowed.add(abs(val))
            # Also add percentage representation (e.g. 0.3471 -> 34.71)
            allowed.add(val * 100.0)
            allowed.add(abs(val * 100.0))

        def walk(obj: Any) -> None:
            if isinstance(obj, (int, float)) and not isinstance(obj, bool):
                add_num(float(obj))
            elif isinstance(obj, dict):
                for v in obj.values():
                    walk(v)
            elif isinstance(obj, (list, tuple)):
                for v in obj:
                    walk(v)

        for ev in registry.all():
            walk(ev.value)
            walk(ev.inputs)
            if ev.sample_size is not None:
                add_num(float(ev.sample_size))

        return sorted(allowed)

    def check_numerical_grounding(
        self,
        text: str,
        registry: EvidenceRegistry,
        rel_tol: float = 0.05,
        abs_tol: float = 0.55,
    ) -> OutputFilterResult:
        """
        Verifies that explicit percentage figures (e.g. '34.7%', '-19.95%') in the text
        match a number present in the deterministic EvidenceRegistry within rounding tolerance.
        """
        allowed_nums = self._collect_allowed_numbers(registry)
        if not allowed_nums:
            return OutputFilterResult(is_clean=True, violations=[])

        violations: List[str] = []
        # Find explicit percentage figures in text like 34.7%, -2.77%, 5.13%
        pct_matches = re.findall(r"([+-]?\d+(?:\.\d+)?)\s*%", text)
        for m in pct_matches:
            try:
                val = abs(float(m))
            except ValueError:
                continue

            matched = False
            for ref in allowed_nums:
                if abs(val - ref) <= max(abs_tol, rel_tol * abs(ref)):
                    matched = True
                    break
            if not matched:
                violations.append(
                    f"Ungrounded percentage figure '{m}%' does not match any Evidence value in registry"
                )

        return OutputFilterResult(
            is_clean=len(violations) == 0,
            violations=violations,
        )
