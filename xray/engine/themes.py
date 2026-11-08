from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from xray.data_loader import DataLoader
from xray.engine.lookthrough import LookThroughEngine, LookThroughResult
from xray.evidence import Evidence


@dataclass
class ThemeIssuerContribution:
    issuer: str
    lookthrough_weight: float
    note: str
    source: str


@dataclass
class ThemeExposure:
    theme_id: str
    name: str
    description: str
    total_weight: float
    contributing_issuers: List[ThemeIssuerContribution] = field(default_factory=list)


@dataclass
class ThemeAnalysisResult:
    themes: List[ThemeExposure]
    dominant_theme: Optional[ThemeExposure]
    non_additive_caveat: str
    evidence_list: List[Evidence] = field(default_factory=list)

    def get_theme_weight(self, theme_id: str) -> float:
        for t in self.themes:
            if t.theme_id == theme_id:
                return t.total_weight
        return 0.0


class ThemeEngine:
    """
    Deterministic economic theme exposure engine.
    Maps decomposed look-through issuer weights onto economic themes defined in data/themes.json.
    Theme exposures are explicitly non-additive across themes because a company can belong
    to multiple macro/supply-chain themes simultaneously.
    """

    def __init__(self, data_loader: Optional[DataLoader] = None) -> None:
        self.loader = data_loader or DataLoader()
        self.themes_data = self.loader.load_themes()

    def analyze(self, lt_result: LookThroughResult) -> ThemeAnalysisResult:
        """
        Computes non-additive economic theme exposures from a LookThroughResult.
        """
        raw_themes = self.themes_data.get("themes", {})
        non_additive_note = self.themes_data.get("metadata", {}).get(
            "non_additive_note",
            "Theme exposures are non-additive across themes because an issuer can belong to multiple themes.",
        )

        theme_exposures: List[ThemeExposure] = []
        evidence_list: List[Evidence] = []

        for theme_id, t_info in raw_themes.items():
            t_name = t_info.get("name", theme_id)
            t_desc = t_info.get("description", "")
            ticker_map: Dict[str, Dict[str, str]] = t_info.get("tickers", {})

            contribs: List[ThemeIssuerContribution] = []
            total_w = 0.0
            sources_set = {"data/themes.json"}

            for issuer, exp in lt_result.issuer_exposures.items():
                if issuer in ticker_map and exp.total_weight > 0:
                    meta = ticker_map[issuer]
                    total_w += exp.total_weight
                    contribs.append(
                        ThemeIssuerContribution(
                            issuer=issuer,
                            lookthrough_weight=round(exp.total_weight, 6),
                            note=meta.get("note", ""),
                            source=meta.get("source", "Company Filing"),
                        )
                    )
                    sources_set.add(meta.get("source", "Company Filing"))

            contribs_sorted = sorted(
                contribs, key=lambda x: x.lookthrough_weight, reverse=True
            )
            rounded_total = round(total_w, 6)

            theme_exp = ThemeExposure(
                theme_id=theme_id,
                name=t_name,
                description=t_desc,
                total_weight=rounded_total,
                contributing_issuers=contribs_sorted,
            )
            theme_exposures.append(theme_exp)

            if rounded_total > 0:
                evidence_list.append(
                    Evidence(
                        id=f"EV-THEME-{theme_id}",
                        metric=f"thematic_exposure_{theme_id.lower()}",
                        value=rounded_total,
                        formula="sum_{i in Theme} w_lookthrough(i)",
                        inputs={
                            "theme_name": t_name,
                            "contributing_issuers": {
                                c.issuer: c.lookthrough_weight for c in contribs_sorted
                            },
                        },
                        data_sources=sorted(sources_set),
                        sample_size=len(contribs_sorted),
                        caveats=[non_additive_note],
                    )
                )

        themes_sorted = sorted(theme_exposures, key=lambda x: x.total_weight, reverse=True)
        dominant = themes_sorted[0] if themes_sorted and themes_sorted[0].total_weight > 0 else None

        return ThemeAnalysisResult(
            themes=themes_sorted,
            dominant_theme=dominant,
            non_additive_caveat=non_additive_note,
            evidence_list=evidence_list,
        )
