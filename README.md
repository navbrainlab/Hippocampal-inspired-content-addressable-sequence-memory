# HICASM

Minimal standalone release of the **Hippocampal-Inspired Content-Addressable
Sequence Memory (HICASM)** model.

The fixed pathway is

$$
R\rightarrow C\rightarrow A\rightarrow B\rightarrow S\rightarrow R.
$$

HICASM uses deterministic episode-level hard winner-take-all in A:

$$
k^*=\arg\max_k D_k,
\qquad D^{\mathrm{WTA}}_{k\ne k^*}=0.
$$

The winning A activity is still projected through the stored A→B association;
the implementation does not directly look up a barcode. The release fixes the
C dynamics, adaptation-based B separator, mature S dynamics, and the
relation-specific modulo-6 Hebbian readout as one canonical model.

## Files

- `hicasm.py` — complete model implementation and small public API.
- `HICASM_demo.ipynb` — initialization, storage, partial-cue retrieval,
  unique-event addressing, ambiguity and disambiguation demonstrations.
- `requirements.txt` — minimal runtime dependencies.

## Quick start

```python
import numpy as np
from hicasm import HICASM, sample_unique_sequences

model = HICASM()
rng = np.random.default_rng(1)
episodes = sample_unique_sequences(rng, 32, 8, 25)
memory = model.build_memory(episodes)

out = model.replay(memory, episodes[:, :3])
exact = np.mean(np.all(out["predicted"] == episodes, axis=1))
print(exact)
```

Initialization performs the frozen independent adaptation and may take several
seconds. All computations use NumPy `float64` on CPU. The demo writes no result
files.

## Reproducibility

The canonical realization uses `model_seed=321001` and
`adaptation_seed=721001`. Default dimensions are
`N_R=32, N_C=160, N_B=640, N_S=1024, H=6`.

This repository contains only the usable frozen model and demonstration. Model
development, rejected variants, historical notebooks and large benchmark
artifacts remain outside this release.
