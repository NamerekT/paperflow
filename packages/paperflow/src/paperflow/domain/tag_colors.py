"""Deterministic tag colours.

Paperless gives every new tag the same washed-out grey, which makes a few
hundred tags indistinguishable at a glance. This picks from a bright,
well-separated palette instead.

The choice is a hash of the name rather than a random draw, and that matters
for two reasons: a tag keeps its colour across runs, and the one-off recolour
of existing tags cannot disagree with what a tag gets when it is created
on the fly during classification. Both callers go through this function.
"""

import hashlib

TAG_COLOR_PALETTE = (
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
    "#469990", "#9a6324", "#800000", "#808000",
    "#000075", "#ff4500", "#daa520", "#2e8b57",
)


def tag_color_for_name(name: str) -> str:
    digest = hashlib.md5(name.strip().lower().encode("utf-8")).hexdigest()
    return TAG_COLOR_PALETTE[int(digest, 16) % len(TAG_COLOR_PALETTE)]
