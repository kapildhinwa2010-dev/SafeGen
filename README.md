# SafeGen

[![CI](https://github.com/kapildhinwa2010-dev/SafeGen/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/SafeGen/actions/workflows/ci.yml)
[![CI](https://github.com/kapildhinwa2010-dev/SafeGen/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/SafeGen/actions/workflows/codeql.yml)

Memory-budgeted matrix generation: estimate the footprint first, raise MemoryBudget if it does not fit.

**Part of the [Kapildhinwa2010-dev Toolkit](https://github.com/kapildhinwa2010-dev/Kapildhinwa2010-dev).**

## What it does

SafeGen owns the SAFE mode of the DNA_DYNIMO generator family:
estimated_bytes() predicts the float64 footprint of a recipe BEFORE
generating; generate_safe() raises MemoryBudget when the estimate
exceeds the allowance; budget_report() gives the numbers without
generating. The HONEST LIMITATION (also in the docstring): the budget
covers the final float64 array only, not NumPy intermediates.

## Install

```bash
pip install -r requirements.txt
```

## Quick start

```python
from safegen import GenRecipe, generate_safe, budget_report, MemoryBudget

recipe = GenRecipe(rows=4096, cols=4096, rank=8, seed=0)
print(budget_report(recipe, max_bytes=8 * 1024 * 1024))
try:
    W = generate_safe(recipe, max_bytes=8 * 1024 * 1024)
except MemoryBudget as e:
    print(e.needed, ">", e.allowed)
```

## API

- GenRecipe(rows, cols, rank, seed)
- estimated_bytes(recipe, dtype_bytes=8)
- generate_safe(recipe, max_bytes) - raises MemoryBudget(needed, allowed)
- budget_report(recipe, max_bytes) - {needed_bytes, max_bytes, fits}

## Honest results and limitations

- The budget counts ONLY the final float64 array; NumPy makes
  intermediate copies during the low-rank product, so real peak
  memory is higher.
- The estimate is rows*cols*dtype arithmetic, not a measurement.

## License

MIT - see [LICENSE](LICENSE).
