"""Readable action labels, and decoding a preflop action sequence.

Pure functions with no dependencies, so they are easy to test. Both the builder
(which precomputes labels into the consolidated JSON) and the loader use them.

A sequence is actions separated by hyphens, in the order they were played:
  "F"     fold
  "C"     call
  "R2.5"  raise to 2.5bb
Six-max positions in order of action: UTG, HJ, CO, BTN, SB, BB.
"""
from __future__ import annotations

POSITIONS = ("UTG", "HJ", "CO", "BTN", "SB", "BB")

# what to call a raise, by how many have come before it
_RAISE_VERB = {1: "Raise", 2: "3-bet", 3: "4-bet", 4: "5-bet"}


def _fmt_size(size: float) -> str:
    """2.5 -> '2.5', 34.0 -> '34', 100.0 -> '100'."""
    return f"{size:g}"


def action_label(action_type: str, size: float | None,
                 prior_raises: int, depth: int) -> str:
    """A short readable description of one action.

    action_type: "fold" | "call" | "raise"
    size:        size in bb (None for fold)
    prior_raises: how many raises came BEFORE this action (open=1, 3-bet=2,
                  ...), which is what makes it a 3-bet, 4-bet or 5-bet
    depth:       stack depth in bb — raise >= depth is all-in
    """
    if action_type == "fold":
        return "Fold"
    if action_type == "call":
        return f"Call {_fmt_size(size)}bb" if size is not None else "Call"
    # raise
    if size is not None and size >= depth * 0.99:
        return f"All-in ({_fmt_size(size)}bb)"
    verb = _RAISE_VERB.get(prior_raises + 1, f"{prior_raises + 1}-bet")
    return f"{verb} to {_fmt_size(size)}bb" if size is not None else verb


def action_key(action_type: str, size: float | None) -> str:
    """A stable key for the action: 'fold', 'call_15bb', 'raise_34bb'."""
    if action_type == "fold":
        return "fold"
    return f"{action_type}_{_fmt_size(size)}bb"


def decode_sequence(sequence: str) -> tuple[list[tuple[str, str]], str | None]:
    """Attach a position to every token, and work out who is left to act.

    Returns (events, hero), where events is [(position, token), ...] in order
    of action and hero is the position to act once the sequence runs out, or
    None if nobody is.

    Tokens arrive in order of action, so a pointer to "the next position that
    has not folded", moving clockwise, is enough. A caller stays in and can
    come round to act again.
    """
    tokens = [t for t in sequence.split("-") if t] if sequence else []
    active = [True] * 6

    def next_active(start: int) -> int | None:
        i = start
        for _ in range(6):
            if active[i]:
                return i
            i = (i + 1) % 6
        return None

    ptr = next_active(0)
    events: list[tuple[str, str]] = []
    for tok in tokens:
        if ptr is None:
            break
        events.append((POSITIONS[ptr], tok))
        if tok == "F":
            active[ptr] = False
        ptr = next_active((ptr + 1) % 6)

    hero = POSITIONS[ptr] if ptr is not None else None
    return events, hero


def _token_word(tok: str) -> str:
    if tok == "F":
        return "fold"
    if tok == "C":
        return "call"
    if tok.startswith("R"):
        return f"raise {tok[1:]}"
    return tok


def describe_line(sequence: str, hero: str | None = None) -> str:
    """A readable action line for a sequence.

    For example "UTG raise 2.5 · CO fold · BTN call · BB raise 15 -> BTN to act".
    A `hero` passed in wins over the decoded one, which covers the spots where
    the real hero comes from is_hero rather than from the sequence.
    """
    events, decoded_hero = decode_sequence(sequence)
    who = hero or decoded_hero
    line = " · ".join(f"{pos} {_token_word(tok)}" for pos, tok in events)
    if who:
        return f"{line} -> {who} to act" if line else f"{who} to act"
    return line
