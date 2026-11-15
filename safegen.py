# SafeGen - Memory-budgeted matrix generation
# Part of the Kapildhinwa2010-dev Toolkit.

"""Generation guarded by an explicit byte budget.

Lineage: DNA_DYNIMO generators. This repo owns the SAFE mode: estimate
the float64 footprint of a recipe BEFORE generating and raise
MemoryBudget when it would exceed the allowance. budget_report() gives
the numbers without generating.

HONEST LIMITATION: the budget covers the final float64 array only - it
does not account for intermediate copies made by NumPy during the low-
rank product, nor for any other allocations in your process.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["GenRecipe", "MemoryBudget", "generate", "generate_safe", "budget_report"]


class MemoryBudget(Exception):
    """A generation request exceeded the allowed byte budget."""

    def __init__(self, needed: int, allowed: int) -> None:
        super().__init__("need {} bytes, budget allows {}".format(needed, allowed))
        self.needed = needed
        self.allowed = allowed


class GenRecipe:
    """A seeded low-rank recipe: rows x cols output from rank x (rows+cols) latents."""

    def __init__(self, rows: int, cols: int, rank: int = 8, seed: int = 0) -> None:
        if rows < 1 or cols < 1:
            raise ValueError("rows and cols must be >= 1")
        if rank < 1:
            raise ValueError("rank must be >= 1")
        self.rows = rows
        self.cols = cols
        self.rank = min(rank, min(rows, cols))
        self.seed = seed

    def _latent(self, seed_offset: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed + seed_offset)
        left = rng.standard_normal((self.rows, self.rank))
        right = rng.standard_normal((self.rank, self.cols))
        return (left @ right) / math.sqrt(self.rank)


def generate(recipe: GenRecipe) -> np.ndarray:
    """The canonical matrix of the recipe (seed + 0)."""
    return recipe._latent(0)

def estimated_bytes(recipe: GenRecipe, dtype_bytes: int = 8) -> int:
    """Float64 (by default) footprint of the recipe's canonical matrix."""
    if dtype_bytes < 1:
        raise ValueError("dtype_bytes must be >= 1")
    return recipe.rows * recipe.cols * dtype_bytes


def generate_safe(recipe: GenRecipe, max_bytes: int = 8 * 1024 * 1024) -> np.ndarray:
    """Generate only if the float64 result fits in ``max_bytes``."""
    if max_bytes < 1:
        raise ValueError("max_bytes must be >= 1")
    needed = estimated_bytes(recipe)
    if needed > max_bytes:
        raise MemoryBudget(needed, max_bytes)
    return generate(recipe)


def budget_report(recipe: GenRecipe, max_bytes: int) -> dict:
    """{"needed_bytes", "max_bytes", "fits"} without generating anything."""
    if max_bytes < 1:
        raise ValueError("max_bytes must be >= 1")
    needed = estimated_bytes(recipe)
    return {"needed_bytes": needed, "max_bytes": max_bytes, "fits": needed <= max_bytes}