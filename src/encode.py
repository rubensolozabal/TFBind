"""Functions to encode DNA grooves for given strands."""

from typing import Mapping, Tuple

from src.constants import MG_funcGroups, mG_funcGroups
from src.utils import ensure_reference_strands
from src.constants import fg_encode_map

__all__ = [
    "groove_encoding_for_strands"
]

def groove_encoding_for_pair(pair: str) -> Tuple[str, str]:
    """
    Return (major_groove, minor_groove) encoding for a Watson–Crick base pair.

    pair: two-letter string in {'AT', 'TA', 'GC', 'CG'} (case-insensitive).
    """
    if not isinstance(pair, str):
        raise TypeError("pair must be a string")
    p = pair.upper()
    if len(p) != 2:
        raise ValueError("pair must be two characters long")

    base1, base2 = p[0], p[1]

    if base1 not in MG_funcGroups or base2 not in MG_funcGroups:
        raise ValueError(f"Unsupported pair {pair!r}")

    major = MG_funcGroups[base1] + MG_funcGroups[base2][::-1]  # Major groove
    minor = mG_funcGroups[base1] + mG_funcGroups[base2][::-1]  # Minor groove

    return major, minor


def groove_encoding_for_strands(
    plus_strand: str,
    minus_strand: str | None = None,
    *,
    complement_map: Mapping[str, str] | None = None,
) -> Tuple[list[str], list[str]]:
    """
    Given plus and minus strands, return per-position major/minor groove encodings.

    Returns:
        (majors, minors): two lists of encodings aligned 5'->3' along the plus strand.
    """
    ref_plus, ref_minus = ensure_reference_strands(
        plus_strand, minus_strand, complement_map=complement_map
    )
    n = len(ref_plus)
    majors: list[str] = []
    minors: list[str] = []
    for i in range(n):
        j = n - 1 - i  # align with the complementary base on the minus strand
        pair = ref_plus[i] + ref_minus[j]
        major, minor = groove_encoding_for_pair(pair)
        majors.append(major)
        minors.append(minor)
    return majors, minors



def one_hot_encode_grooves(
    major: list[list[str]] | list[str],
    minor: list[list[str]] | list[str],
    *,
    encode_map: dict[str, list[int]] | None = None,
) -> tuple[list[list[list[int]]], list[list[list[int]]]]:
    """
    Given major and minor groove encodings grouped per position (L x 4),
    return L x 4 x OH one-hot encodings using fg_encode_map.

    Accepts each position as:
      - a 4-char string (e.g., "ADAM"), or
      - an iterable of 4 single-character symbols (e.g., ['A','D','A','M']).
    """
    if not isinstance(major, list) or not isinstance(minor, list):
        raise TypeError("major and minor must be lists grouped per position (L x 4)")

    emap = fg_encode_map if encode_map is None else encode_map
    if not isinstance(emap, dict):
        raise TypeError("encode_map must be a dict-like mapping")

    def to_symbols(chunk) -> list[str]:
        if isinstance(chunk, str):
            return list(chunk)
        return [c if isinstance(c, str) else str(c) for c in chunk]

    def encode_groups(groups: list) -> list[list[list[int]]]:
        out: list[list[list[int]]] = []
        for chunk in groups:
            symbols = to_symbols(chunk)
            encoded_chunk: list[list[int]] = []
            for ch in symbols:
                key = ch if ch in emap else ch.upper()
                if key not in emap:
                    raise ValueError(f"Unsupported functional group symbol {ch!r}")
                encoded_chunk.append(list(emap[key]))
            out.append(encoded_chunk)
        return out

    return encode_groups(major), encode_groups(minor)

def categorical_encode_grooves(
    major: list[list[str]] | list[str],
    minor: list[list[str]] | list[str],
    *,
    encode_map: dict[str, list[int]] | None = None,
    categories: dict[str, int] | None = None,
) -> tuple[list[list[int]], list[list[int]]]:
    """
    Given major and minor groove encodings grouped per position (L x 4),
    return L x 4 integer encodings (categorical IDs).

    If `categories` is not provided, it is derived from `encode_map`
    (defaults to fg_encode_map) by taking the index of the hot bit.
    Accepts each position as:
        - a 4-char string (e.g., "ADAM"), or
        - an iterable of 4 single-character symbols (e.g., ['A','D','A','M']).
    """
    if not isinstance(major, list) or not isinstance(minor, list):
        raise TypeError("major and minor must be lists grouped per position (L x 4)")

    emap = fg_encode_map if encode_map is None else encode_map
    if categories is None:
        if not isinstance(emap, dict):
            raise TypeError("encode_map must be a dict-like mapping")
        cat_map: dict[str, int] = {}
        for k, v in emap.items():
            vec = list(v)
            ones = [i for i, x in enumerate(vec) if int(x) == 1]
            if len(ones) != 1:
                raise ValueError(f"encode_map for key {k!r} is not one-hot")
            cat_map[k] = ones[0]
    else:
        if not isinstance(categories, dict):
            raise TypeError("categories must be a dict-like mapping")
        cat_map = categories

    def to_symbols(chunk) -> list[str]:
        if isinstance(chunk, str):
            return list(chunk)
        return [c if isinstance(c, str) else str(c) for c in chunk]

    def encode_groups(groups: list) -> list[list[int]]:
        out: list[list[int]] = []
        for chunk in groups:
            symbols = to_symbols(chunk)
            encoded_chunk: list[int] = []
            for ch in symbols:
                key = ch if ch in cat_map else ch.upper()
                if key not in cat_map:
                    raise ValueError(f"Unsupported functional group symbol {ch!r}")
                encoded_chunk.append(int(cat_map[key]))
            out.append(encoded_chunk)
        return out

    return encode_groups(major), encode_groups(minor)


import numpy as np


_FG_VECTORS = {
    key: np.asarray(value, dtype=np.float32)
    for key, value in fg_encode_map.items()
}


def _normalise_symbol(symbol: str) -> str:
    if not isinstance(symbol, str) or not symbol:
        raise ValueError(f"Invalid functional group symbol {symbol!r}")
    if symbol in _FG_VECTORS:
        return symbol
    upper = symbol.upper()
    if upper in _FG_VECTORS:
        return upper
    raise KeyError(f"Symbol {symbol!r} is not present in fg_encode_map")


def groove_stack_to_tensor(
    major: 'Iterable[Iterable[str]]',
    minor: 'Iterable[Iterable[str]]',
    length: int | None = None,
) -> np.ndarray:
    major_seq = list(major)
    minor_seq = list(minor)

    if length is None:
        length = len(major_seq)

    if len(major_seq) != length or len(minor_seq) != length:
        raise ValueError(
            f"Expected {length} positions, got {len(major_seq)} (major) "
            f"and {len(minor_seq)} (minor)"
        )

    encoded = np.zeros((4, 8, length), dtype=np.float32)
    for pos in range(length):
        major_chunk = list(major_seq[pos])
        minor_chunk = list(minor_seq[pos])

        if len(major_chunk) != 4 or len(minor_chunk) != 4:
            raise ValueError("Each groove position must contain 4 functional groups")

        for row_idx, symbol in enumerate(major_chunk):
            try:
                vec = _FG_VECTORS[_normalise_symbol(symbol)]
            except KeyError as exc:
                raise ValueError(f"Unsupported major groove symbol {symbol!r}") from exc
            encoded[:, row_idx, pos] = vec

        for row_idx, symbol in enumerate(minor_chunk):
            try:
                vec = _FG_VECTORS[_normalise_symbol(symbol)]
            except KeyError as exc:
                raise ValueError(f"Unsupported minor groove symbol {symbol!r}") from exc
            encoded[:, row_idx + 4, pos] = vec

    return encoded
