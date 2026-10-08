# Storage-Capacity Scenario Model for a Dual-Plane GB300 NVL72 Cluster Network

Companion code for:

> Liang, R. (2026). *A Conservative Scenario Method for Storage-Access Capacity
> and Failure Bounds in a Dual-Plane GB300 NVL72 Cluster Network* (design-study
> preprint). DOI: [10.5281/zenodo.23247527](https://doi.org/10.5281/zenodo.23247527)

This repository contains the capacity model exactly as used to produce the
scenario tables of the paper, the scenario definitions (S01–S14d and the HOLD
project scenario P00), the unit-test suite that pins the model's semantics, and
a reproduction script that rebuilds the paper's supplementary dataset and
verifies it against the published copy.

```
python reproduce.py
OK: 19 scenario inputs, 19 result records and 16 figure rows match the published dataset
```

## What the model does

The model checks storage-access demand against a dual-plane north-south
(BF3/DPU) fabric, per cut, per side and per direction:

- **Cuts checked:** client port, NS-leaf uplink, NS aggregate uplink, and the
  storage attachment itself (CVS ports for SA-1/SA-2, NS-leaf ports for SA-3).
- **Conservative direction:** where an input is unknown, the model computes a
  bound, never a point estimate presented as fact. A demand that exceeds a
  bound may still be a HOLD (insufficient input), but a demand is never cleared
  in error by the bounding step.
- **Status discipline:** every scenario and every input is labeled FACT,
  DESIGN, STUDY or HOLD, and HOLD inputs gate the verdict (see `HoldGate`
  tests). No value in this repository records an executed test on hardware.

## Layout

| File | Content |
|---|---|
| `storage_model.py` | The capacity model: flows, scenarios, cut checks, degraded modes, bounds |
| `scenarios.py` | Scenario definitions S01–S14d and P00, titles as published (Table IS2-4) |
| `ports_chart.py` | Per-bar data extraction behind the published port figure |
| `reproduce.py` | Rebuilds the dataset and compares it with `dataset/` (the published copy) |
| `tests/test_models.py` | 60 unit tests pinning units, saturation, directionality, degraded modes, bounds and HOLD gating |
| `dataset/` | The supplementary dataset as published (CC-BY-4.0) |

Run the tests with `python -m pytest tests/` (or `python -m unittest discover tests`).
No dependencies beyond the Python 3 standard library.

## Related resources

- The 75-page network reference design the scenarios are checked against:
  [kkdatasvc.com/lab/downloads](https://www.kkdatasvc.com/lab/downloads/) —
  sheets N-101 … N-951 with an interactive viewer.
- Water-use study of the same reference data center:
  DOI [10.5281/zenodo.23245321](https://doi.org/10.5281/zenodo.23245321).
- Citable design numbers index:
  [kkdatasvc.com/lab/numbers](https://www.kkdatasvc.com/lab/numbers/).

## License

- Code (`*.py`): MIT License (see `LICENSE`).
- `dataset/`: CC-BY-4.0, © 2026 K&K Data Service Inc., as published with the paper.

## Citation

If you use this model, cite the paper (see `CITATION.cff`):

> Liang, R. (2026). A Conservative Scenario Method for Storage-Access Capacity
> and Failure Bounds in a Dual-Plane GB300 NVL72 Cluster Network.
> Zenodo. https://doi.org/10.5281/zenodo.23247527
