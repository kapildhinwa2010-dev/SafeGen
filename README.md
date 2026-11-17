# SafeGen

SafeGen is a small Python library for generating seeded, dense low-rank NumPy matrices while optionally checking that the estimated size of the final result fits a caller-supplied byte budget. The project metadata declares version `0.1.0` and Python `>=3.9`. The repository contains a single-module implementation and focused tests; it does not establish production support or a published release process.

## Features

- `GenRecipe(rows, cols, rank=8, seed=0)` describes a matrix recipe. Dimensions and rank must be positive; the effective rank is capped at the smaller dimension.
- `generate(recipe)` creates the recipe's matrix as a NumPy `float64` array using seeded random factors and a low-rank matrix product.
- `estimated_bytes(recipe, dtype_bytes=8)` computes `rows * cols * dtype_bytes`; the default assumes eight bytes per output element.
- `generate_safe(recipe, max_bytes=8 * 1024 * 1024)` checks that estimate before generating. It raises `MemoryBudget` with `needed` and `allowed` attributes if the estimate exceeds the limit.
- `budget_report(recipe, max_bytes)` reports `needed_bytes`, `max_bytes`, and `fits` without generating a matrix.

## Installation

Install from a local checkout with Python and pip:

```bash
python -m pip install .
```

NumPy is the runtime dependency (`numpy>=1.24`). The package metadata requires Python 3.9 or newer. SafeGen provides a Python import API; no command-line executable is defined.

## Usage

```python
from safegen import GenRecipe, MemoryBudget, budget_report, generate_safe

recipe = GenRecipe(rows=64, cols=48, rank=4, seed=0)
max_bytes = 64 * 48 * 8  # estimated size of the float64 result

print(budget_report(recipe, max_bytes=max_bytes))

try:
    matrix = generate_safe(recipe, max_bytes=max_bytes)
    print(matrix.shape)
except MemoryBudget as exc:
    print(f"Estimated result is {exc.needed} bytes; budget is {exc.allowed} bytes")
```

`estimated_bytes` is also available from `safegen` when callers need the estimate directly. `generate` does not perform a budget check; use `generate_safe` when that check is required.

## Tests and CI

The test suite was run locally against the checked-out repository commit with:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
```

Result: **10 passed**. The GitHub Actions workflow defines pytest, compilation, and an import smoke test on Python 3.10, 3.11, and 3.12; the workflow result for this commit was not checked. The CodeQL workflow is configured in the repository, but its result was not checked either.

## Limitations and scope

The budget is an estimate of the final output array only. It does not include the temporary NumPy factor arrays or intermediates created during multiplication, allocations elsewhere in the process, or total peak memory. Passing the check is therefore not a guarantee that generation will fit within a process or machine's actual memory limit. The implementation generates the full dense result in memory; it does not provide streaming, persistence, a CLI, or other application interfaces.

No benchmark or measured performance result is provided by the repository or this review.

## License

The repository includes an MIT License. See [LICENSE](LICENSE) for the exact terms and copyright notice.
