import ast
from typing import Mapping, Sequence, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from src.constants import COLOR_FG, COMPLEMENT_BASE


__all__ = [
    "reverse_complement",
    "ensure_reference_strands",
    "apply_strand_mutation",
    "categorize_change",
    "plot_groove_functional_groups",
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
        "U": "dUTP",
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
        return "mismatch"
    if len(non_wc) == 1:
        return non_wc[0]
    return "mod_on_both_bases"


def _coerce_groove_matrix(value: Sequence[Sequence[str]] | str) -> np.ndarray:
    """Return a 2-D numpy array for the groove representation."""
    if isinstance(value, str):
        value = ast.literal_eval(value)
    array = np.asarray(value, dtype=object)
    if array.ndim != 2:
        raise ValueError("Groove inputs must be two-dimensional (positions x features)")
    return array


def _strand_to_list(strand: Sequence[str] | str | None) -> list[str] | None:
    """Return a list of bases for plus/minus strands."""
    if strand is None:
        return None
    if isinstance(strand, str):
        return list(strand)
    return [str(base) for base in strand]


def plot_groove_functional_groups(
    groove_major: Sequence[Sequence[str]] | str,
    groove_minor: Sequence[Sequence[str]] | str,
    *,
    plus_strand: Sequence[str] | str | None = None,
    minus_strand: Sequence[str] | str | None = None,
    column_labels: Sequence[str] | None = None,
    major_label_prefix: str = "M",
    minor_label_prefix: str = "m",
    color_map: Mapping[str, Tuple[float, float, float]] | None = None,
    legend_keys: Sequence[str] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Visualise concatenated major/minor groove functional groups as a grid."""
    major = _coerce_groove_matrix(groove_major)
    minor = _coerce_groove_matrix(groove_minor)
    if major.shape[0] != minor.shape[0]:
        raise ValueError("Groove matrices must have the same number of rows (positions)")

    groove_concat = np.concatenate([major, minor], axis=1)
    n_rows, n_cols = groove_concat.shape
    colors = color_map or COLOR_FG
    fallback = colors.get("x", (0.0, 0.0, 0.0))
    color_img = np.array(
        [
            [colors.get(str(val), fallback) for val in row]
            for row in groove_concat
        ],
        dtype=float,
    )
    if column_labels is None:
        column_labels = [
            *(f"{major_label_prefix}_{i+1}" for i in range(major.shape[1])),
            *(f"{minor_label_prefix}_{i+1}" for i in range(minor.shape[1])),
        ]
    if len(column_labels) != n_cols:
        raise ValueError("column_labels must match the concatenated groove width")

    plus_list = _strand_to_list(plus_strand)
    minus_list = _strand_to_list(minus_strand)
    if plus_list is not None and len(plus_list) != n_rows:
        raise ValueError("plus_strand must match the number of groove rows")
    if minus_list is not None and len(minus_list) != n_rows:
        raise ValueError("minus_strand must match the number of groove rows")

    if plus_list is not None and minus_list is not None:
        pair_labels = [f"{p}{m}" for p, m in zip(plus_list, reversed(minus_list))]
    elif plus_list is not None:
        pair_labels = plus_list
    elif minus_list is not None:
        pair_labels = list(reversed(minus_list))
    else:
        pair_labels = [f"{idx + 1}" for idx in range(n_rows)]

    if ax is None:
        height = max(6.0, n_rows * 1.5)
        fig, ax = plt.subplots(figsize=(5, height))
    else:
        fig = ax.figure

    ax.imshow(color_img, aspect="equal", origin="upper", interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.6)
    ax.set_axisbelow(False)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(column_labels, fontsize=10, rotation=45)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(pair_labels, fontsize=12)

    legend_order = legend_keys or [
        key for key in ("A", "D", "M", "n", "s", "x") if key in colors
    ]
    if legend_order:
        handles = [Patch(color=colors[key], label=key) for key in legend_order]
        ax.legend(
            handles=handles,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
        )

    fig.tight_layout()
    return fig
