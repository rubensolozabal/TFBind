from typing import Mapping, Tuple

from src.constants import COMPLEMENT_BASE


__all__ = [
    "reverse_complement",
    "ensure_reference_strands",
    "apply_strand_mutation",
]


def reverse_complement(
    sequence: str, *, complement_map: Mapping[str, str] | None = None
) -> str:
    """Return the reverse-complement of a nucleotide sequence."""
    if not isinstance(sequence, str):
        raise TypeError("sequence must be a string")
    complement = complement_map or COMPLEMENT_BASE
    try:
        return "".join(complement[base] for base in reversed(sequence))
    except KeyError as exc:
        raise ValueError(f"Unknown base {exc.args[0]!r} in sequence") from exc


def ensure_reference_strands(
    plus_strand: str | None,
    minus_strand: str | None = None,
    *,
    complement_map: Mapping[str, str] | None = None,
) -> Tuple[str, str]:
    """Normalise plus/minus reference strands, filling in complements if needed."""
    if not isinstance(plus_strand, str):
        raise TypeError("plus_strand must be provided as a string")

    complement = complement_map or COMPLEMENT_BASE
    reference_minus = (
        minus_strand
        if isinstance(minus_strand, str)
        else reverse_complement(plus_strand, complement_map=complement)
    )

    if len(plus_strand) != len(reference_minus):
        raise ValueError("plus_strand and minus_strand must be the same length")

    return plus_strand, reference_minus


def apply_strand_mutation(
    change: str,
    position: int,
    plus_strand: str,
    minus_strand: str | None = None,
    *,
    complement_map: Mapping[str, str] | None = None,
) -> Tuple[str, str]:
    """Return mutated plus/minus strands after applying a 1-based change."""
    if not isinstance(change, str) or not change:
        raise ValueError("change must be a non-empty string")

    if not isinstance(position, int):
        raise TypeError("position must be an integer")

    complement = complement_map or COMPLEMENT_BASE
    ref_plus, ref_minus = ensure_reference_strands(
        plus_strand, minus_strand, complement_map=complement
    )

    sequence_length = len(ref_plus)
    if not 1 <= position <= sequence_length:
        raise ValueError(f"position must be within [1, {sequence_length}]")

    plus_index = position - 1
    minus_index = sequence_length - position

    plus_base = change[0]
    if len(change) > 1:
        minus_base = change[1]
    else:
        try:
            minus_base = complement[plus_base]
        except KeyError as exc:
            raise ValueError(
                f"No complement mapping available for base {plus_base!r}"
            ) from exc

    plus_chars = list(ref_plus)
    minus_chars = list(ref_minus)
    plus_chars[plus_index] = plus_base
    minus_chars[minus_index] = minus_base
    return "".join(plus_chars), "".join(minus_chars)


def categorize_change(change: str) -> str:
    mapping = {
        "A": "WC",
        "C": "WC",
        "G": "WC",
        "T": "WC",
        "U": "dUPT",
        "X": "5mC",
        "a": "7dA",
        "g": "7dG",
        "I": "I",
        "M": "6mA",
        "D": "D",
    }
    chars = list(str(change))
    labels = [mapping.get(ch, mapping.get(str(ch).upper(), "WC")) for ch in chars]
    non_wc = [lbl for lbl in labels if lbl != "WC"]
    if not non_wc:
        return "none"
    if len(non_wc) == 1:
        return non_wc[0]
    return "both"
