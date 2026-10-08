"""Reproduce the published scenario dataset from the model source.

Rebuilds the scenario inputs and results exactly as published in the
supplementary dataset of DOI 10.5281/zenodo.23247527 and compares them
against the copies in dataset/.  Exit code 0 means every scenario input
and every numeric result matches the published record.

Usage:  python reproduce.py
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ports_chart
import scenarios
import storage_model as sm

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(HERE, "dataset")


def compare(path, a, b, errors):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                errors.append(f"{path}.{k}: present on one side only")
                continue
            compare(f"{path}.{k}", a[k], b[k], errors)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            errors.append(f"{path}: length {len(a)} != {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            compare(f"{path}[{i}]", x, y, errors)
    elif isinstance(a, float) or isinstance(b, float):
        ok = (a is None) == (b is None) and a is not None and \
            math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
        if not ok:
            errors.append(f"{path}: {a!r} != {b!r}")
    elif a != b:
        errors.append(f"{path}: {a!r} != {b!r}")


def main():
    built = scenarios.build()
    inputs = {s["id"]: s for s in (sm.scenario_to_dict(sc) for sc in built)}
    results = {sc.id: sm.evaluate(sc) for sc in built}

    pub_in = json.load(open(os.path.join(DS, "storage_scenarios_inputs.json"), encoding="utf-8"))
    pub_res = json.load(open(os.path.join(DS, "storage_scenarios_results.json"), encoding="utf-8"))

    errors = []
    for rec in pub_in["scenarios"]:
        sid = rec["id"]
        if sid not in inputs:
            errors.append(f"inputs:{sid}: not produced by scenarios.build()")
            continue
        compare(f"inputs:{sid}", rec, inputs[sid], errors)
    for rec in pub_res:
        sid = rec["id"]
        if sid not in results:
            errors.append(f"results:{sid}: not produced by scenarios.build()")
            continue
        compare(f"results:{sid}", rec, results[sid], errors)

    pub_fig = json.load(open(os.path.join(DS, "figure_2_2_data.json"), encoding="utf-8"))
    fig_rows = ports_chart.ports_chart_data([results[sc.id] for sc in built])
    by_id = {sc.id: sc.title for sc in built}
    for row in fig_rows:
        row["title"] = by_id[row["id"]]
    compare("figure_2_2", pub_fig, fig_rows, errors)

    n_in = len(pub_in["scenarios"])
    n_res = len(pub_res)
    if errors:
        print(f"MISMATCH: {len(errors)} differences against the published dataset")
        for e in errors[:40]:
            print(" ", e)
        return 1
    print(f"OK: {n_in} scenario inputs, {n_res} result records and "
          f"{len(fig_rows)} figure rows match the published dataset")
    return 0


if __name__ == "__main__":
    sys.exit(main())
