"""Builds the consolidated preflop dataset.

Reads the raw per-node JSON files and extracts from each:
  - the CORRECT hero from raw.players_info[is_hero] (not from meta.position),
  - the strategy per hand, as raw range-weighted frequencies,
  - the available actions with readable labels precomputed,
  - a corrected description and the decoded action line,
  - quality flags (label_mismatch, range_fraction),
and writes them into a single slim `data/nl100_100bb.json` — about 9% of the
original size, with the `raw` field dropped — so nothing at runtime depends on
where the source nodes came from.

Run it after re-downloading data:
    python -m pokersolver.ranges.gto.build [SOURCE_DIR] [OUT_PATH]
The source directory comes from `PF_GTO_SOURCE_DIR`, or ./raw_nodes.
Pass OUT_PATH explicitly for another family or depth (a 2bb open, say), or you
will overwrite
baseline `nl100_100bb.json`.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from pokersolver.ranges.hand_grid import all_hand_classes
from .labels import action_key, action_label, describe_line
from .loader import DATA_DIR, LOW_COVERAGE

# Where the raw nodes live. This used to be derived from path depth
# (`parents[6]`), which tied it to wherever the repo happened to sit — one
# directory move and the build quietly found nothing.
DEFAULT_SOURCE = Path(os.environ.get("PF_GTO_SOURCE_DIR")
                      or Path.cwd() / "raw_nodes")
OUT_PATH = DATA_DIR / "nl100_100bb.json"

_HANDS = all_hand_classes()
_COMBOS = {h: (6 if len(h) == 2 else (4 if h.endswith("s") else 12)) for h in _HANDS}
_TOTAL_COMBOS = sum(_COMBOS.values())


def _hero_from_raw(raw: dict[str, Any], meta_pos: str) -> str:
    """The real hero from is_hero, falling back to meta.position."""
    for pi in raw.get("players_info", []) or []:
        player = pi.get("player") or {}
        if player.get("is_hero"):
            return player.get("position") or meta_pos
    return meta_pos


def _flatten_strategy(processed: dict[str, Any]
                      ) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    """Merge the raise/call/fold maps into {hand: {action_key: freq}} plus a catalogue.

    The catalogue is ordered raises first (largest down to smallest), then
    calls, then fold — so it reads the way a viewer shows it, most aggressive
    on the left.
    """
    raise_sizes = sorted(processed.get("raise", {}), key=lambda s: float(s[:-2]),
                         reverse=True)
    call_sizes = sorted(processed.get("call", {}), key=lambda s: float(s[:-2]),
                        reverse=True)
    catalog: list[dict[str, Any]] = []
    for size in raise_sizes:
        catalog.append({"type": "raise", "size": float(size[:-2]),
                        "key": action_key("raise", float(size[:-2]))})
    for size in call_sizes:
        catalog.append({"type": "call", "size": float(size[:-2]),
                        "key": action_key("call", float(size[:-2]))})
    catalog.append({"type": "fold", "size": None, "key": "fold"})

    strategy: dict[str, dict[str, float]] = {}
    for size in raise_sizes:
        for hand, freq in processed["raise"][size].items():
            if freq:
                strategy.setdefault(hand, {})[action_key("raise", float(size[:-2]))] = freq
    for size in call_sizes:
        for hand, freq in processed["call"][size].items():
            if freq:
                strategy.setdefault(hand, {})[action_key("call", float(size[:-2]))] = freq
    for hand, freq in processed.get("fold", {}).items():
        if freq:
            strategy.setdefault(hand, {})["fold"] = freq
    return strategy, catalog


def _ev_map(processed: dict[str, Any]) -> dict[str, float]:
    ev = processed.get("ev") or {}
    if isinstance(ev.get("main"), dict):
        return {h: v for h, v in ev["main"].items() if v is not None}
    # fallback keys (raise_/call_): take the first map
    for v in ev.values():
        if isinstance(v, dict):
            return {h: x for h, x in v.items() if x is not None}
    return {}


def _range_fraction(strategy: dict[str, dict[str, float]]) -> float:
    in_range = sum(_COMBOS[h] for h, f in strategy.items()
                   if h in _COMBOS and sum(f.values()) > 1e-6)
    return in_range / _TOTAL_COMBOS


def build_spot(spot_id: str, doc: dict[str, Any]) -> dict[str, Any]:
    meta = doc["meta"]
    processed = doc.get("processed") or {}
    depth = int(meta.get("depth", 100))
    sequence = meta.get("preflopActions", "")
    meta_pos = meta.get("position", "")
    hero = _hero_from_raw(doc.get("raw") or {}, meta_pos)

    strategy, catalog = _flatten_strategy(processed)
    prior_raises = sequence.count("R")
    actions = [
        {**a, "label": action_label(a["type"], a["size"], prior_raises, depth)}
        for a in catalog
    ]
    frac = _range_fraction(strategy)
    return {
        "hero": hero,
        "position_meta": meta_pos,
        "label": f"{hero} to act — {meta.get('description', spot_id)}",
        "line": describe_line(sequence, hero),
        "sequence": sequence,
        "actions": actions,
        "strategy": strategy,
        "ev": _ev_map(processed),
        "range_fraction": round(frac, 4),
        "label_mismatch": hero != meta_pos,
    }


def build(source_dir: Path, out_path: Path = OUT_PATH) -> dict[str, Any]:
    files = sorted(source_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"no data in {source_dir}")
    spots: dict[str, Any] = {}
    skipped: list[str] = []
    for fp in files:
        doc = json.loads(fp.read_text(encoding="utf-8"))
        if not doc.get("processed"):
            skipped.append(fp.stem)
            continue
        spots[fp.stem] = build_spot(fp.stem, doc)

    mismatches = [i for i, s in spots.items() if s["label_mismatch"]]
    low_cov = sum(1 for s in spots.values() if s["range_fraction"] < LOW_COVERAGE)
    out = {
        "meta": {
            "gametype": "Cash6mGeneral_6mNL100R25",
            "depth": 100,
            "open_size": 2.5,
            "source": "gto_downloader/tests/data/nl100_100bb",
            "n_spots": len(spots),
            "n_label_mismatch": len(mismatches),
            "n_low_coverage": low_cov,
            "note": "Frequencies are range-weighted (raw). Use the loader "
                    "GtoSpot.decision() pro normalizaci. Hrdina je z is_hero.",
        },
        "spots": spots,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8")
    print(f"wrote {len(spots)} spots -> {out_path}")
    print(f"  skipped (no processed data): {len(skipped)}"
          + (f" → {skipped}" if skipped else ""))
    print(f"  label_mismatch (hero != meta.position): {len(mismatches)}")
    print(f"  narrow spots (<{LOW_COVERAGE:.0%} of range): {low_cov}")
    size_kb = out_path.stat().st_size / 1024
    print(f"  velikost artefaktu: {size_kb:.0f} kB")
    return out


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SOURCE
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT_PATH
    build(src, out)
