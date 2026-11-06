# Python API

## Dependency-light calibration

```python
import numpy as np

from cpo_stcm.evaluation import conformal_p_values

reference = np.array([0.1, 0.2, 0.3])
p_values = conformal_p_values(reference, np.array([0.05, 0.4]))
```

## Conditional conformal helpers

```python
from cpo_stcm.conformal import mondrian_conformal_p_values

p_values = mondrian_conformal_p_values(
    calibration_scores=calibration,
    calibration_strata=calibration_strata,
    test_scores=test,
    test_strata=test_strata,
)
```

## Semantic evidence

`cpo_stcm.static_evidence` provides slot-to-candidate mapping, adaptive
calibration, weighted isotonic scoring, and serialized detection conversion.

## Temporal models

The following symbols are lazily resolved and require the `ml` extra:

```python
from cpo_stcm import LSTMSequenceAutoencoder, TemporalWindowDataset
```

Importing `cpo_stcm` alone does not import PyTorch.

## Data API

```python
from cpo_stcm.data import (
    fetch_cpo_led_bench,
    verify_archive,
    build_packed_array,
)
```

All download and packing functions raise documented exceptions rather than
silently accepting size, hash, shape, or path-validation failures.
