"""Per-bar data of the published port figure (figure_2_2_data.json).

Pure data extraction from evaluate() results; the plotting code of the
design record is not included here.
"""
from math import ceil


def _np_ports(tbps):
    return ceil(tbps / 0.4 - 1e-9) if tbps > 1e-12 else 0


def ports_chart_data(results):
    """One bar pair per CVS-port scenario, every value taken from one side and one direction.

    Side: normal and NS-leaf loss -> the side/direction with the highest C4 utilisation (first on a tie);
    CVS loss -> the impaired side; side loss -> the surviving side.  No-path port-equivalents:
    CVS loss -> that direction's unserved share, which is all on the impaired side; side loss -> the lost
    side's unrecoverable share of that direction (kind "lost_side"); NS-leaf loss -> that direction's
    unserved upper bound, shown only on the failed side.  pool = ports available for storage on that side.
    """
    rows = []
    for r in results:
        if r["result_state"] != "EVALUATED_STUDY" or r["attachment"] != "cvs_ports":
            continue
        kind, _, fs = r["failure"].partition(":")
        cand = [(d, s, c) for d, v in r["directions"].items() for s, c in v["sides"].items()
                if v["line_tbps"] > 0]
        if kind == "cvs_lost":
            cand = [x for x in cand if x[1] == fs]
        best = max(cand, key=lambda x: (x[2]["storage_attachment"]["utilisation"], x[2]["storage_ports_required"]))
        d, s, c = best
        un = r["directions"][d]["unserved_line_tbps"]
        if kind == "ns_leaf_lost" and s != fs:
            un = 0.0
        if kind == "normal":
            un = 0.0
        twin = None
        if kind == "normal":
            o = "B" if s == "A" else "A"
            oc = r["directions"][d]["sides"].get(o)
            twin = oc is not None and all(oc[k] == c[k] for k in ("storage_ports_required", "storage_ports_assigned",
                                                                   "ports_available_for_storage"))
        rows.append({"id": r["id"], "side": s, "direction": d, "required": c["storage_ports_required"],
                     "nopath_ports": _np_ports(un), "nopath_tbps": un,
                     "nopath_kind": "" if un <= 1e-12 else ("lost_side" if kind == "side_lost" else "impaired_side"),
                     "assigned": c["storage_ports_assigned"], "pool": c["ports_available_for_storage"],
                     "both_sides_equal": bool(twin)})
    return rows
