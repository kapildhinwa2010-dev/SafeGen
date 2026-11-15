import numpy as np
import pytest

from safegen import GenRecipe, MemoryBudget, budget_report, generate, generate_safe


def test_generate_shape():
    recipe = GenRecipe(rows=16, cols=32, rank=4, seed=0)
    assert generate(recipe).shape == (16, 32)


def test_generate_deterministic():
    recipe = GenRecipe(rows=8, cols=8, seed=3)
    assert np.array_equal(generate(recipe), generate(recipe))


def test_recipe_rejects_bad_args():
    with pytest.raises(ValueError):
        GenRecipe(rows=0, cols=8)


def test_safe_allows_small():
    recipe = GenRecipe(rows=4, cols=4)
    assert generate_safe(recipe, max_bytes=1024).shape == (4, 4)


def test_safe_raises_memory_budget():
    recipe = GenRecipe(rows=1024, cols=1024)
    with pytest.raises(MemoryBudget) as exc_info:
        generate_safe(recipe, max_bytes=1024)
    assert exc_info.value.needed == 1024 * 1024 * 8
    assert exc_info.value.allowed == 1024


def test_safe_rejects_bad_budget():
    with pytest.raises(ValueError):
        generate_safe(GenRecipe(rows=2, cols=2), max_bytes=0)


# --- budget reporting (SafeGen-specific) ---


def test_budget_report_fits():
    report = budget_report(GenRecipe(rows=10, cols=10), max_bytes=8000)
    assert report == {"needed_bytes": 800, "max_bytes": 8000, "fits": True}


def test_budget_report_does_not_fit():
    report = budget_report(GenRecipe(rows=1024, cols=1024), max_bytes=1024)
    assert report["fits"] is False


def test_budget_report_does_not_generate():
    # An oversized recipe must still be reportable (no allocation happens)
    recipe = GenRecipe(rows=10**6, cols=10**6)
    report = budget_report(recipe, max_bytes=1024)
    assert report["fits"] is False


def test_budget_report_rejects_bad_budget():
    with pytest.raises(ValueError):
        budget_report(GenRecipe(rows=2, cols=2), max_bytes=0)