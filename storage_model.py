"""Storage attachment capacity model for a dual-plane GB300 NVL72 cluster's north-south (BF3/DPU) fabric.

Unit conventions (decimal SI, used everywhere in this module):
  1 TB    = 1e12 byte          1 GB/s = 1e9 byte/s
  1 Tb/s  = 1e12 bit/s         1 GB/s = 8 Gb/s = 0.008 Tb/s
Capacities are per direction.  A full-duplex link of 400 Gb/s offers 400 Gb/s
in each direction.  At client ports and on SA-1/SA-2 paths read (storage -> client)
and write (client -> storage) use opposite physical directions and are checked
separately.  On SA-3 (storage on NS-leaf ports) a hosting leaf sends reads upward
while its clients send writes upward, so read and write share each physical
direction of the leaf uplinks; those cuts are evaluated per physical direction.

Client accounting is per flow group: each flow names the client trays it spans
(clients) and optionally a client group label.  Flows with the same label share the
same trays; the relation between different groups is "unknown" (worst case: nested
groups, loads add on one tray) unless the scenario declares them "disjoint".
Unrelated flows and opposite-direction flows never dilute a hot client's demand.

NS-leaf loss removes one leaf's tray slots in total (54), not 54 per group.  Tray placement is
unknown, so the lost load is bounded (heaviest placement / load forced past the other 29 leaves);
each cut is judged on its own conservative bound and unserved demand and stretch are carried as bounds.

Verdicts are three-state: within / exceeds / HOLD.  A cut whose upper-bound demand is
within capacity is "within"; whose lower-bound demand exceeds capacity is "exceeds";
anything between, or any cut depending on an absent input, is HOLD.

Status vocabulary:  FACT / DESIGN values come from the P03b baseline summary
(fabric_summary.csv); STUDY values are illustrative; HOLD inputs are absent
(None) and block any within/exceeds statement for the affected cut.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from math import ceil, isfinite
from typing import Optional

TB = 1e12               # byte
GB = 1e9                # byte
BITS_PER_BYTE = 8
EPS = 1e-12


def tb_to_bits(tb: float) -> float:
    return tb * TB * BITS_PER_BYTE


def gbytes_s_to_tbps(gbytes_s: float) -> float:
    return gbytes_s * GB * BITS_PER_BYTE / 1e12


def tbps_to_gbytes_s(tbps: float) -> float:
    return tbps * 1e12 / BITS_PER_BYTE / GB


def frame_efficiency(payload_bytes: float, overhead_bytes: float) -> float:
    """Payload fraction of line rate for a given per-frame overhead (both inputs HOLD per vendor)."""
    if payload_bytes <= 0 or overhead_bytes < 0:
        raise ValueError("payload must be > 0 and overhead >= 0")
    return payload_bytes / (payload_bytes + overhead_bytes)


def _num(name: str, v, lo: Optional[float] = None, hi: Optional[float] = None,
         lo_open: bool = False, hi_open: bool = False, integer: bool = False):
    """Reject malformed numbers: non-numeric, bool, non-finite or outside the stated domain."""
    if v is None:
        return
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{name} must be a number")
    if not isfinite(v):
        raise ValueError(f"{name} must be finite")
    if integer and float(v) != int(v):
        raise ValueError(f"{name} must be an integer")
    if lo is not None and (v < lo or (lo_open and v == lo)):
        raise ValueError(f"{name} = {v} below its domain")
    if hi is not None and (v > hi or (hi_open and v == hi)):
        raise ValueError(f"{name} = {v} above its domain")


# ---------------------------------------------------------------- baseline (DESIGN, P03b)
@dataclass(frozen=True)
class NSBaseline:
    leaves_per_side: int = 30
    leaf_down_ports: int = 54
    leaf_up_ports: int = 28
    cvs_per_side: int = 7
    cvs_unused_ports: int = 8
    port_gbps: float = 400.0
    trays: int = 1620
    leaf_free_ports: int = 46        # 128 - 54 - 28, reported unused, not reserved

    @property
    def port_tbps(self) -> float:
        return self.port_gbps / 1000.0

    @property
    def uplink_tbps_per_side(self) -> float:          # 336 Tb/s per direction
        return self.leaves_per_side * self.leaf_up_ports * self.port_tbps

    @property
    def leaf_uplink_tbps(self) -> float:              # 11.2 Tb/s per leaf per direction
        return self.leaf_up_ports * self.port_tbps

    @property
    def cvs_unused_ports_per_side(self) -> int:       # 56
        return self.cvs_per_side * self.cvs_unused_ports

    @property
    def cvs_unused_tbps_per_side(self) -> float:      # 22.4 Tb/s per direction
        return self.cvs_unused_ports_per_side * self.port_tbps

    @property
    def trays_per_leaf(self) -> int:
        return self.leaf_down_ports


BASE = NSBaseline()

FLOW_KINDS = {
    "dataset_staging": "read",      # storage -> local cache
    "training_read": "read",        # streaming reads not served by local cache
    "restart_read": "read",         # checkpoint reload after failure / requeue
    "checkpoint_write": "write",
    "management": None,             # in-band services: carried in reserve_fraction
    "scheduler_observation": None,  # telemetry: carried in reserve_fraction / OOB
}
REJECTED_KINDS = {"collective", "all_reduce", "allreduce", "compute_fabric"}
FAILURES = ("normal", "side_lost", "cvs_lost", "ns_leaf_lost")
CVS_LOSS_ASSUMPTIONS = ("even_spread_rehome", "even_spread_no_rehome")
SA3_PLACEMENTS = ("uniform",)


@dataclass
class Flow:
    name: str
    kind: str
    unique_tb: Optional[float] = None       # unique bytes moved (TB)
    fanout: float = 1.0                     # copies of the same bytes delivered without de-duplication
    cache_hit: float = 0.0                  # fraction served from local cache (reads only)
    deadline_s: Optional[float] = None
    rate_gbytes_s: Optional[float] = None   # alternative to size/deadline (payload GB/s, aggregate)
    clients: Optional[int] = None           # client trays this flow spans
    group: Optional[str] = None             # client group label; None = a group of its own

    def __post_init__(self):
        if self.kind in REJECTED_KINDS:
            raise ValueError("compute-fabric collective traffic is not storage demand")
        if self.kind not in FLOW_KINDS:
            raise ValueError(f"unknown flow kind {self.kind}")
        _num("cache_hit", self.cache_hit, 0.0, 1.0)
        _num("fanout", self.fanout, 1.0)
        _num("unique_tb", self.unique_tb, 0.0)
        _num("rate_gbytes_s", self.rate_gbytes_s, 0.0)
        _num("deadline_s", self.deadline_s, 0.0, lo_open=True)
        _num("clients", self.clients, 1, BASE.trays, integer=True)
        if self.direction == "write" and self.cache_hit != 0.0:
            raise ValueError("cache_hit applies to reads only")
        if self.rate_gbytes_s is not None and (self.unique_tb is not None or self.deadline_s is not None):
            raise ValueError("give either rate_gbytes_s or unique_tb/deadline_s, not both")

    @property
    def direction(self) -> Optional[str]:
        return FLOW_KINDS[self.kind]

    @property
    def group_key(self) -> str:
        return self.group if self.group is not None else f"flow:{self.name}"

    def missing(self) -> list[str]:
        m = []
        if self.rate_gbytes_s is None:
            if self.unique_tb is None:
                m.append(f"{self.name}.unique_tb")
            if self.deadline_s is None:
                m.append(f"{self.name}.deadline_s")
        if self.clients is None:
            m.append(f"{self.name}.clients")
        return m

    def wire_tb(self) -> Optional[float]:
        """Bytes crossing the network (TB): unique x fan-out x (1 - cache hit)."""
        if self.unique_tb is None:
            return None
        return self.unique_tb * self.fanout * (1.0 - self.cache_hit)

    def payload_tbps(self) -> Optional[float]:
        if self.rate_gbytes_s is not None:
            return gbytes_s_to_tbps(self.rate_gbytes_s) * (1.0 - self.cache_hit)
        w = self.wire_tb()
        if w is None or self.deadline_s is None:
            return None
        return tb_to_bits(w) / self.deadline_s / 1e12


@dataclass
class Scenario:
    id: str
    title: str
    status: str                              # STUDY | HOLD
    flows: list
    attachment: str = "cvs_ports"            # cvs_ports (SA-1/SA-2) | leaf_distributed (SA-3)
    payload_efficiency: Optional[float] = None
    reserve_fraction: Optional[float] = None  # share of client / leaf / CVS links held for other traffic
    storage_ports: dict = field(default_factory=lambda: {"A": None, "B": None})
    border_reserved_ports: Optional[int] = None   # CVS unused ports held for upstream/border, per side
    topology_delta: Optional[str] = None          # explicit, unselected delta that adds ports outside the baseline pool
    leaf_storage_ports: Optional[int] = None      # SA-3: storage ports per hosting NS leaf (<= 46)
    sa3_hosting_leaves: Optional[int] = None      # SA-3: hosting NS leaves per side (<= 30)
    sa3_placement: Optional[str] = None           # SA-3: "uniform" (STUDY) or None (HOLD)
    side_split: dict = field(default_factory=lambda: {"A": 0.5, "B": 0.5})
    group_relation: str = "unknown"          # unknown (nested worst case) | disjoint (declared)
    failure: str = "normal"                  # normal | side_lost:A | cvs_lost:A | ns_leaf_lost:A
    redirect: Optional[float] = None         # side_lost / ns_leaf_lost: share of the failed portion carried by the other side
    cvs_loss_assumption: Optional[str] = None  # cvs_lost: even_spread_rehome | even_spread_no_rehome; None = HOLD
    clients_per_leaf_max: Optional[int] = None
    storage_read_limit_gbytes_s: Optional[float] = None    # storage system, HOLD unless supplied
    storage_write_limit_gbytes_s: Optional[float] = None
    host_port_achievable_gbps: Optional[float] = None      # client achievable per BF3 port, HOLD
    note: str = ""
    base: NSBaseline = field(default_factory=NSBaseline)


def _validate(sc: Scenario):
    """Domain checks that apply whatever inputs are still HOLD.  Malformed data raises."""
    b = sc.base
    if sc.status not in ("STUDY", "HOLD"):
        raise ValueError("status must be STUDY or HOLD")
    if sc.attachment not in ("cvs_ports", "leaf_distributed"):
        raise ValueError("attachment must be cvs_ports or leaf_distributed")
    if set(sc.side_split) != {"A", "B"}:
        raise ValueError("side_split needs exactly A and B")
    for s, v in sc.side_split.items():
        _num(f"side_split.{s}", v, 0.0, 1.0)
    if abs(sum(sc.side_split.values()) - 1.0) > 1e-9:
        raise ValueError("side_split must sum to 1")
    _num("payload_efficiency", sc.payload_efficiency, 0.0, 1.0, lo_open=True)
    _num("reserve_fraction", sc.reserve_fraction, 0.0, 1.0, hi_open=True)
    _num("redirect", sc.redirect, 0.0, 1.0)
    _num("clients_per_leaf_max", sc.clients_per_leaf_max, 1, b.trays_per_leaf, integer=True)
    _num("border_reserved_ports", sc.border_reserved_ports, 0, b.cvs_unused_ports_per_side, integer=True)
    _num("leaf_storage_ports", sc.leaf_storage_ports, 0, b.leaf_free_ports, integer=True)
    _num("sa3_hosting_leaves", sc.sa3_hosting_leaves, 1, b.leaves_per_side, integer=True)
    for k in ("storage_read_limit_gbytes_s", "storage_write_limit_gbytes_s", "host_port_achievable_gbps"):
        _num(k, getattr(sc, k), 0.0, lo_open=True)
    if sc.host_port_achievable_gbps is not None and sc.host_port_achievable_gbps > b.port_gbps:
        raise ValueError("host_port_achievable_gbps above port line rate")
    if set(sc.storage_ports) != {"A", "B"}:
        raise ValueError("storage_ports needs exactly A and B")
    for s, v in sc.storage_ports.items():
        _num(f"storage_ports.{s}", v, 0, integer=True)
        if v is not None and sc.border_reserved_ports is not None and not sc.topology_delta:
            if v > b.cvs_unused_ports_per_side - sc.border_reserved_ports:
                raise ValueError(f"storage_ports.{s} = {v} exceeds baseline CVS pool "
                                 f"{b.cvs_unused_ports_per_side} - k_border {sc.border_reserved_ports}; "
                                 "use an explicit topology_delta STUDY")
    if sc.topology_delta is not None and (not str(sc.topology_delta).strip() or sc.attachment != "cvs_ports"):
        raise ValueError("topology_delta must be a non-empty description on a cvs_ports scenario")
    if sc.group_relation not in ("unknown", "disjoint"):
        raise ValueError("group_relation must be unknown or disjoint")
    kind, _, side = sc.failure.partition(":")
    if kind not in FAILURES or (kind == "normal") != (side == "") or (side and side not in ("A", "B")):
        raise ValueError(f"unknown failure {sc.failure}")
    if kind == "cvs_lost" and sc.redirect is not None:
        raise ValueError("cvs_lost uses cvs_loss_assumption, not redirect")
    if sc.cvs_loss_assumption is not None and sc.cvs_loss_assumption not in CVS_LOSS_ASSUMPTIONS:
        raise ValueError("unknown cvs_loss_assumption")
    if sc.sa3_placement is not None and sc.sa3_placement not in SA3_PLACEMENTS:
        raise ValueError("unsupported sa3_placement")
    groups = {}
    for f in sc.flows:
        if f.direction and f.clients is not None:
            if groups.setdefault(f.group_key, f.clients) != f.clients:
                raise ValueError(f"flows of group {f.group_key} must span the same client count")
    if sc.group_relation == "disjoint" and sum(groups.values()) > b.trays:
        raise ValueError("disjoint client groups exceed installed trays")
    if sc.clients_per_leaf_max is not None:
        room = b.leaves_per_side * sc.clients_per_leaf_max
        if any(n > room for n in groups.values()) or (sc.group_relation == "disjoint" and sum(groups.values()) > room):
            raise ValueError(f"client groups do not fit {b.leaves_per_side} leaves x clients_per_leaf_max "
                             f"{sc.clients_per_leaf_max}")


def _missing(sc: Scenario) -> list[str]:
    m = []
    for f in sc.flows:
        m += f.missing()
    for k in ("payload_efficiency", "reserve_fraction", "clients_per_leaf_max"):
        if getattr(sc, k) is None:
            m.append(k)
    kind = sc.failure.partition(":")[0]
    if sc.attachment == "cvs_ports":
        for s in ("A", "B"):
            if sc.storage_ports.get(s) is None:
                m.append(f"storage_ports.{s}")
        if sc.border_reserved_ports is None:
            m.append("border_reserved_ports")
    else:
        for k in ("leaf_storage_ports", "sa3_hosting_leaves", "sa3_placement"):
            if getattr(sc, k) is None:
                m.append(k)
        if kind != "normal":
            # hosted storage is lost with its leaf/side; served share depends on the storage
            # system's replica placement, which this model does not represent
            m.append("sa3_failure_replica_placement")
    if kind in ("side_lost", "ns_leaf_lost") and sc.redirect is None:
        m.append("redirect")
    if kind == "cvs_lost" and sc.cvs_loss_assumption is None:
        m.append("cvs_loss_assumption")
    return m


def _bounded(lo: float, hi: float, capacity: float, basis: str) -> dict:
    """Cut with lower/upper demand bounds (Tb/s) against capacity (Tb/s)."""
    if capacity <= 0:
        u = float("inf") if hi > EPS else 0.0
        ulo = float("inf") if lo > EPS else 0.0
    else:
        u, ulo = hi / capacity, lo / capacity
    if u <= 1.0 + EPS:
        v = "within"
    elif ulo > 1.0 + EPS:
        v = "exceeds"
    else:
        v = "HOLD"
    return {"demand_tbps": hi, "demand_lower_tbps": lo, "capacity_tbps": capacity,
            "utilisation": u, "verdict": v, "basis": basis}


def _cut(demand: float, capacity: float, basis: str = "exact") -> dict:
    return _bounded(demand, demand, capacity, basis)


def _per_client(groups: list, relation: str, trays: int = BASE.trays, pigeonhole: bool = True) -> tuple:
    """(lower, upper, basis) of the hottest client's load.  groups: [(clients, per-client Tb/s)].

    Unknown overlap: upper = all groups on one tray; lower = the largest single group, raised by
    pairwise pigeonhole (two groups with n_i + n_j > trays must share a tray).  Overlaps forced only
    by three or more groups are not used, so the lower bound stays conservative (possible false HOLD).
    """
    gs = [(n, g) for n, g in groups if g > 0]
    loads = [g for _, g in gs]
    if not loads:
        return 0.0, 0.0, "exact"
    if len(loads) == 1:
        return loads[0], loads[0], "exact"
    if relation == "disjoint":
        return max(loads), max(loads), "exact (disjoint groups)"
    lo = max(loads)
    for i in range(len(gs) if pigeonhole else 0):
        for j in range(i + 1, len(gs)):
            if gs[i][0] + gs[j][0] > trays:
                lo = max(lo, gs[i][1] + gs[j][1])
    return lo, sum(loads), "envelope (group overlap unknown)"


def _leaf_loss(groups: list, relation: str, slots: int, leaves: int) -> tuple:
    """(lower, upper, forced) load of one failed leaf.  groups: [(clients, per-client Tb/s on that side)].

    The failed leaf holds at most `slots` trays in total, whatever the number of groups.  The other
    leaves hold at most (leaves - 1) x slots, so trays beyond that are forced onto the failed leaf.
    Upper: heaviest placement (unknown overlap: groups nested on the same trays, each contributes
    per-client x min(n, slots); disjoint or single group: the heaviest `slots` trays).  Lower: load
    that must sit on the failed leaf (unknown overlap: groups nested so that each forces only
    max(0, n - room) trays; disjoint or single group: the lightest max(0, N - room) trays).
    forced[i] is True when group i has a tray behind the failed leaf in every placement.
    """
    room = (leaves - 1) * slots
    forced = [n > room for n, _ in groups]
    if relation == "disjoint" or len(groups) == 1:
        trays = sorted(((g, n) for n, g in groups), reverse=True)
        def fill(order, k):
            tot = 0.0
            for g, n in order:
                take = min(n, k)
                tot += take * g
                k -= take
                if k <= 0:
                    break
            return tot
        total_n = sum(n for n, _ in groups)
        hi = fill(trays, slots)
        lo = fill(trays[::-1], max(0, total_n - room))
    else:
        hi = sum(g * min(n, slots) for n, g in groups)
        lo = sum(g * max(0, n - room) for n, g in groups)
    return lo, hi, forced


def _per_leaf(groups: list, relation: str, slots: int, side_total: float, leaves: int) -> tuple:
    """(lower, upper, basis) of the most loaded leaf's client demand.

    Upper bound: worst placement of client trays onto one leaf of `slots` trays.  Unknown group
    relation: nested groups, each contributes per-client x min(n, slots).  Disjoint groups:
    fractional fill of the slots in descending per-client load.  Lower bound: the side total
    spread evenly over all leaves (some leaf carries at least the average).
    """
    gs = [(n, g) for n, g in groups if g > 0]
    lo = side_total / leaves
    if not gs:
        return 0.0, 0.0, "exact"
    if relation == "disjoint" or len(gs) == 1:
        left, hi = slots, 0.0
        for n, g in sorted(gs, key=lambda x: -x[1]):
            take = min(n, left)
            hi += take * g
            left -= take
            if left <= 0:
                break
    else:
        hi = sum(g * min(n, slots) for n, g in gs)
    return min(lo, hi), hi, "worst-case tray placement envelope"


def evaluate(sc: Scenario) -> dict:
    _validate(sc)
    b = sc.base
    missing = _missing(sc)
    out = {"id": sc.id, "title": sc.title, "status": sc.status, "attachment": sc.attachment,
           "failure": sc.failure, "missing_inputs": missing, "directions": {}, "physical_cuts": {},
           "flags": [], "note": sc.note}
    if missing or sc.status == "HOLD":
        out["result_state"] = "NOT_EVALUATED_HOLD"
        return out
    if sc.topology_delta:
        pool = b.cvs_unused_ports_per_side - sc.border_reserved_ports
        if any(sc.storage_ports[s] > pool for s in ("A", "B")):
            out["flags"].append(f"assigned ports exceed baseline CVS pool {pool}: evaluated as unselected "
                                f"topology delta ({sc.topology_delta})")

    usable = 1.0 - sc.reserve_fraction
    slots = min(sc.clients_per_leaf_max, b.trays_per_leaf)
    fail_kind, _, fs = sc.failure.partition(":")
    other = {"A": "B", "B": "A"}
    out["result_state"] = "EVALUATED_STUDY"
    sa3 = sc.attachment == "leaf_distributed"
    side_totals = {}

    for d in ("read", "write"):
        flows = [f for f in sc.flows if f.direction == d]
        payload = sum(f.payload_tbps() for f in flows)
        line = payload / sc.payload_efficiency
        grp = {}
        for f in flows:
            n, l = grp.get(f.group_key, (f.clients, 0.0))
            grp[f.group_key] = (n, l + f.payload_tbps() / sc.payload_efficiency)
        groups = list(grp.values())                       # [(clients, group line Tb/s)]
        sigma = dict(sc.side_split)
        side_dem = {s: line * sigma[s] for s in ("A", "B")}
        # per-client factor: share of a client's group line carried by side s (after failure);
        # cfac_lo[s][i] is the factor group i certainly carries (lower bound of the hottest client)
        cfac = dict(sigma)
        cfac_lo = {s: [sigma[s]] * len(groups) for s in ("A", "B")}
        cfac_all = {s: list(v) for s, v in cfac_lo.items()}  # factors every tray of the group carries
        unserved = 0.0
        # placement extremes for ns_leaf_lost: (side A, side B, unserved) with the least / most
        # load behind the failed leaf; for every other state both equal the exact result
        extremes = None
        side_lo = None
        leaves_s = {s: b.leaves_per_side for s in ("A", "B")}
        up_cap = {s: b.uplink_tbps_per_side * usable for s in ("A", "B")}
        leaf_cap = {s: b.leaf_uplink_tbps * usable for s in ("A", "B")}
        client_cap = {s: (sc.host_port_achievable_gbps or b.port_gbps) / 1000.0 * usable for s in ("A", "B")}
        sp = ({s: sc.storage_ports[s] for s in ("A", "B")} if not sa3
              else {s: sc.leaf_storage_ports for s in ("A", "B")})
        avail = {s: (b.cvs_unused_ports_per_side - sc.border_reserved_ports) if not sa3 else b.leaf_free_ports
                 for s in ("A", "B")}

        if fail_kind == "side_lost":
            o = other[fs]
            moved = side_dem[fs] * sc.redirect
            unserved = side_dem[fs] - moved
            side_dem[o] += moved
            side_dem[fs] = 0.0
            cfac[o] = sigma[o] + sigma[fs] * sc.redirect
            cfac[fs] = 0.0
            cfac_lo = cfac_all = {o: [cfac[o]] * len(groups), fs: [0.0] * len(groups)}
            up_cap[fs] = leaf_cap[fs] = client_cap[fs] = 0.0
            sp[fs] = 0
        elif fail_kind == "cvs_lost":
            up_cap[fs] *= (b.cvs_per_side - 1) / b.cvs_per_side
            leaf_cap[fs] *= (b.leaf_up_ports - b.leaf_up_ports // b.cvs_per_side) / b.leaf_up_ports
            k = sp[fs]
            lost_ports = ceil(k / b.cvs_per_side)          # ports spread evenly; worst CVS holds the ceiling
            avail[fs] -= ceil(avail[fs] / b.cvs_per_side)
            sp[fs] = k - lost_ports
            if sc.cvs_loss_assumption == "even_spread_no_rehome" and k > 0:
                lost = side_dem[fs] * lost_ports / k
                unserved = lost
                side_dem[fs] -= lost
        elif fail_kind == "ns_leaf_lost":
            o = other[fs]
            rho = sc.redirect
            pcf = [(n, gl * sigma[fs] / n) for n, gl in groups]   # per-client load on the failed side
            lost_lo, lost_hi, forced = _leaf_loss(pcf, sc.group_relation, slots, b.leaves_per_side)
            lost_lo, lost_hi = min(lost_lo, side_dem[fs]), min(lost_hi, side_dem[fs])
            ext = []
            for lost in (lost_lo, lost_hi):
                sd = {fs: side_dem[fs] - lost, o: side_dem[o] + lost * rho}
                ext.append((sd["A"], sd["B"], lost * (1.0 - rho)))
            extremes = ext
            # each cut is judged on its own conservative bound: the failed side keeps at most
            # side - lost_lo, the other side receives at most rho x lost_hi; only moved trays add load
            side_lo = {fs: side_dem[fs] - lost_hi, o: side_dem[o] + lost_lo * rho}
            side_dem = {fs: side_dem[fs] - lost_lo, o: side_dem[o] + lost_hi * rho}
            unserved = lost_hi * (1.0 - rho)
            cfac[o] = sigma[o] + sigma[fs] * rho            # any tray may sit behind the failed leaf
            cfac_lo[o] = [sigma[o] + (sigma[fs] * rho if f else 0.0) for f in forced]
            up_cap[fs] *= (b.leaves_per_side - 1) / b.leaves_per_side
            leaves_s[fs] = b.leaves_per_side - 1

        if side_lo is None:
            side_lo = dict(side_dem)
        if extremes is None:
            extremes = [(side_dem["A"], side_dem["B"], unserved)] * 2
        side_totals[d] = dict(side_dem)
        # served/unserved reported for the placement with the most unserved demand (conservative);
        # conservation is checked separately for both placement extremes
        served = line - unserved
        unserved_lo = extremes[0][2]
        cuts = {}
        for s in ("A", "B"):
            if fail_kind == "side_lost" and s == fs:
                continue
            pc = [(n, gl * cfac[s] / n) for n, gl in groups]
            pc_lo = [(n, gl * f / n) for (n, gl), f in zip(groups, cfac_lo[s])]
            _, c_hi, c_basis = _per_client(pc, sc.group_relation, b.trays)
            # forced-group factors hold only for trays behind the failed leaf, so the pigeonhole
            # (shared tray) bound uses the factors every tray carries
            pc_all = [(n, gl * f / n) for (n, gl), f in zip(groups, cfac_all[s])]
            c_lo = max(_per_client(pc_lo, sc.group_relation, b.trays, pigeonhole=False)[0],
                       _per_client(pc_all, sc.group_relation, b.trays)[0])
            if c_lo < c_hi - EPS and c_basis.startswith("exact"):
                c_basis = "envelope (redirected share only on trays behind the failed leaf)"
            cuts[s] = {"client_port": _bounded(c_lo, c_hi, client_cap[s], c_basis)}
            exact = side_lo[s] >= side_dem[s] - EPS
            dem_basis = "exact" if exact else "placement bounds (trays behind the failed leaf)"
            if not sa3:
                l_lo, l_hi, l_basis = _per_leaf(pc, sc.group_relation, slots, side_lo[s], leaves_s[s])
                cuts[s]["ns_leaf_uplink"] = _bounded(l_lo, l_hi, leaf_cap[s], l_basis)
                cuts[s]["ns_aggregate_uplink"] = _bounded(side_lo[s], side_dem[s], up_cap[s], dem_basis)
                cuts[s]["storage_attachment"] = _bounded(side_lo[s], side_dem[s], sp[s] * b.port_tbps, dem_basis)
                port_req = ceil(side_dem[s] / b.port_tbps - 1e-9) if side_dem[s] > EPS else 0
                cuts[s]["storage_ports_required_lower"] = (ceil(side_lo[s] / b.port_tbps - 1e-9)
                                                           if side_lo[s] > EPS else 0)
            else:
                hosted = side_dem[s] / sc.sa3_hosting_leaves   # per hosting leaf, uniform placement
                cuts[s]["storage_attachment"] = _cut(hosted, sp[s] * b.port_tbps, "per hosting leaf (uniform placement)")
                port_req = ceil(hosted / b.port_tbps - 1e-9) if hosted > EPS else 0
                cuts[s]["_client_leaf"] = _per_leaf(pc, sc.group_relation, slots, side_dem[s], b.leaves_per_side)
            cuts[s]["storage_ports_required"] = port_req
            cuts[s]["storage_ports_assigned"] = sp[s]
            cuts[s]["ports_available_for_storage"] = avail[s]
            unit = "per hosting leaf " if sa3 else ""
            if port_req > sp[s]:
                out["flags"].append(f"{d}/{s}: storage ports required {unit}{port_req} > assigned {sp[s]}")
            if port_req > avail[s]:
                where = "NS-leaf unused ports" if sa3 else "CVS unused ports available"
                out["flags"].append(f"{d}/{s}: required {unit}{port_req} > {where} {avail[s]} "
                                    "-> topology delta outside this port pool (unselected)")
            if cuts[s]["client_port"]["verdict"] != "within":
                out["flags"].append(f"{d}/{s}: client_port {cuts[s]['client_port']['verdict']} "
                                    f"(hottest client {cuts[s]['client_port']['utilisation']:.3f})")

        payload_gb = tbps_to_gbytes_s(payload)
        lim = sc.storage_read_limit_gbytes_s if d == "read" else sc.storage_write_limit_gbytes_s
        endpoint = {"demand_gbytes_s": payload_gb, "limit_gbytes_s": lim,
                    "verdict": "HOLD" if lim is None else ("within" if payload_gb <= lim * (1 + EPS) else "exceeds")}
        out["directions"][d] = {
            "payload_tbps": payload, "line_tbps": line, "payload_gbytes_s": payload_gb,
            "served_line_tbps": served, "unserved_line_tbps": unserved,
            "served_line_upper_tbps": line - unserved_lo, "unserved_line_lower_tbps": unserved_lo,
            "conservation_ok": all(abs(a + bb + u - line) <= 1e-9 * max(1.0, line) for a, bb, u in extremes)
            and abs(served + unserved - line) <= 1e-9 * max(1.0, line),
            "sides": cuts, "storage_endpoint": endpoint,
        }
        if unserved > EPS:
            rng = f"{unserved_lo:.3f}..{unserved:.3f}" if unserved_lo < unserved - EPS else f"{unserved:.3f}"
            out["flags"].append(f"{d}: {rng} Tb/s line-rate demand without a path ({_why(sc)})")

    if sa3:
        _sa3_physical(sc, out, side_totals, usable)
    else:
        for d, v in out["directions"].items():
            for s, c in v["sides"].items():
                for n in ("ns_leaf_uplink", "ns_aggregate_uplink", "storage_attachment"):
                    if c[n]["verdict"] != "within":
                        out["flags"].append(f"{d}/{s}: {n} {c[n]['verdict']} -> port/topology delta")

    for d, v in out["directions"].items():
        cutlist = [c[n] for c in v["sides"].values()
                   for n in ("client_port", "ns_leaf_uplink", "ns_aggregate_uplink", "storage_attachment") if n in c]
        worst = max((c["utilisation"] for c in cutlist), default=0.0)
        worst_lo = max((c["demand_lower_tbps"] / c["capacity_tbps"] if c["capacity_tbps"] > 0 else c["utilisation"]
                        for c in cutlist), default=0.0)
        v["worst_network_utilisation"] = worst
        v["deadline_stretch_lower"] = None
        if v["unserved_line_tbps"] > EPS:
            v["deadline_stretch"] = None
            v["deadline_note"] = ("not applicable: part of the demand has no path; deadline not met for that share"
                                  if v["unserved_line_lower_tbps"] > EPS else
                                  "not determined: depending on tray placement part of the demand may have no path")
        elif any(c["verdict"] == "HOLD" for c in cutlist):
            v["deadline_stretch"] = None
            v["deadline_note"] = "not determined: a cut is HOLD"
        else:
            # every cut is within or exceeds on both bounds: the network-only stretch lies between the
            # largest lower-bound and the largest upper-bound utilisation (equal when all cuts are exact)
            v["deadline_stretch"] = max(1.0, worst)
            v["deadline_stretch_lower"] = max(1.0, worst_lo)
            v["deadline_note"] = ("network-only stretch" + (" (upper bound; lower bound in deadline_stretch_lower)"
                                                            if v["deadline_stretch_lower"] < v["deadline_stretch"] - EPS else "")
                                  + "; storage system (C5) and host limits not included")
    return out


def _why(sc):
    if sc.redirect is not None:
        return f"redirect={sc.redirect:g}"
    return sc.cvs_loss_assumption or ""


def _sa3_physical(sc: Scenario, out: dict, side_totals: dict, usable: float):
    """SA-3 leaf-uplink cuts per physical direction (uniform placement, normal state only).

    A hosting leaf sends hosted reads up (leaf -> CVS) and receives hosted writes down; its client
    trays receive reads down and send writes up.  Remote fraction of a client's traffic under
    uniform placement is (1 - 1/30).  Aggregate per physical direction:
    up = down = (R + W) x (1 - 1/30).  Per leaf, conservative (no local deduction):
    up <= R/H + W_clients(leaf),  down <= W/H + R_clients(leaf).
    """
    b = sc.base
    H = sc.sa3_hosting_leaves
    remote = 1.0 - 1.0 / b.leaves_per_side
    for s in ("A", "B"):
        R, W = side_totals["read"][s], side_totals["write"][s]
        rs = out["directions"]["read"]["sides"][s]
        ws = out["directions"]["write"]["sides"][s]
        r_lo, r_hi, _ = rs.pop("_client_leaf")
        w_lo, w_hi, _ = ws.pop("_client_leaf")
        agg = (R + W) * remote
        cap_agg = b.uplink_tbps_per_side * usable
        cap_leaf = b.leaf_uplink_tbps * usable
        up = _bounded(max(R / H, w_lo) * remote, R / H + w_hi, cap_leaf,
                      "physical leaf->CVS: hosted reads out + client writes out (envelope)")
        down = _bounded(max(W / H, r_lo) * remote, W / H + r_hi, cap_leaf,
                        "physical CVS->leaf: hosted writes in + client reads in (envelope)")
        out["physical_cuts"][s] = {
            "ns_aggregate_up": _cut(agg, cap_agg, "physical leaf->CVS, read+write remote share"),
            "ns_aggregate_down": _cut(agg, cap_agg, "physical CVS->leaf, read+write remote share"),
            "ns_leaf_up": up, "ns_leaf_down": down,
            "logical_read_tbps": R, "logical_write_tbps": W,
        }
        # the read/write rows show the physical cut each logical direction loads most
        rs["ns_aggregate_uplink"] = ws["ns_aggregate_uplink"] = out["physical_cuts"][s]["ns_aggregate_up"]
        worst_leaf = up if up["utilisation"] >= down["utilisation"] else down
        rs["ns_leaf_uplink"] = ws["ns_leaf_uplink"] = worst_leaf
        for n in ("ns_aggregate_up", "ns_aggregate_down", "ns_leaf_up", "ns_leaf_down"):
            c = out["physical_cuts"][s][n]
            if c["verdict"] != "within":
                out["flags"].append(f"physical/{s}: {n} {c['verdict']} -> port/topology delta")
        for d, side in (("read", rs), ("write", ws)):
            if side["storage_attachment"]["verdict"] != "within":
                out["flags"].append(f"{d}/{s}: storage_attachment {side['storage_attachment']['verdict']} per hosting leaf")


def required_ports(line_tbps: float, port_gbps: float = 400.0, reserve: float = 0.0) -> int:
    """Smallest port count n with n x port x (1 - reserve) >= line-rate demand."""
    _num("line_tbps", line_tbps, 0.0)
    _num("port_gbps", port_gbps, 0.0, lo_open=True)
    _num("reserve", reserve, 0.0, 1.0, hi_open=True)
    cap = port_gbps / 1000.0 * (1 - reserve)
    return 0 if line_tbps <= 0 else ceil(line_tbps / cap - 1e-9)


def scenario_to_dict(sc: Scenario) -> dict:
    d = asdict(sc)
    d.pop("base")
    return d
