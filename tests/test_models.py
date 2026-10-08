import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scenarios  # noqa: E402
import storage_model as sm  # noqa: E402


def study(flows, **kw):
    p = dict(payload_efficiency=1.0, reserve_fraction=0.0, clients_per_leaf_max=54,
             border_reserved_ports=0, storage_ports={"A": 10, "B": 10})
    p.update(kw)
    return sm.Scenario(id="T", title="t", status=p.pop("status", "STUDY"), flows=flows, **p)


def rcut(r, d="read", s="A", n="client_port"):
    return r["directions"][d]["sides"][s][n]


class Units(unittest.TestCase):
    def test_tb_per_second_is_8_tbps(self):
        f = sm.Flow("x", "checkpoint_write", unique_tb=1.0, deadline_s=1.0, clients=1)
        self.assertAlmostEqual(f.payload_tbps(), 8.0)

    def test_gbytes_tbps_roundtrip(self):
        self.assertAlmostEqual(sm.gbytes_s_to_tbps(125.0), 1.0)
        self.assertAlmostEqual(sm.tbps_to_gbytes_s(336.0), 42000.0)

    def test_baseline_values(self):
        b = sm.BASE
        self.assertAlmostEqual(b.uplink_tbps_per_side, 336.0)
        self.assertAlmostEqual(b.cvs_unused_tbps_per_side, 22.4)
        self.assertEqual(b.cvs_unused_ports_per_side, 56)
        self.assertAlmostEqual(b.leaf_uplink_tbps, 11.2)
        self.assertAlmostEqual(54 / 28, 1.9285714, places=6)

    def test_efficiency_raises_line_rate(self):
        r = sm.evaluate(study([sm.Flow("w", "checkpoint_write", rate_gbytes_s=500, clients=1620)],
                              payload_efficiency=0.8))
        self.assertAlmostEqual(r["directions"]["write"]["line_tbps"], 5.0)

    def test_frame_efficiency(self):
        self.assertAlmostEqual(sm.frame_efficiency(900, 100), 0.9)


class Saturation(unittest.TestCase):
    def test_storage_ports_saturate(self):
        # 10 ports x 0.4 = 4 Tb/s per side; 10 Tb/s read split 5/5 -> 1.25
        r = sm.evaluate(study([sm.Flow("r", "training_read", rate_gbytes_s=1250, clients=1620)]))
        a = r["directions"]["read"]["sides"]["A"]
        self.assertAlmostEqual(a["storage_attachment"]["utilisation"], 1.25)
        self.assertEqual(a["storage_attachment"]["verdict"], "exceeds")
        self.assertEqual(a["storage_ports_required"], 13)
        self.assertTrue(any("required 13 > assigned 10" in f for f in r["flags"]))

    def test_topology_delta_threshold(self):
        r = sm.evaluate(study([sm.Flow("r", "training_read", rate_gbytes_s=6000, clients=1620)],
                              storage_ports={"A": 48, "B": 48}, border_reserved_ports=8))
        self.assertTrue(any("topology delta outside this port pool" in f for f in r["flags"]))

    def test_exact_capacity_is_within(self):
        r = sm.evaluate(study([sm.Flow("r", "training_read", rate_gbytes_s=1000, clients=1620)]))
        self.assertEqual(r["directions"]["read"]["sides"]["A"]["storage_attachment"]["verdict"], "within")

    def test_reserve_reduces_shared_capacity(self):
        r = sm.evaluate(study([sm.Flow("r", "training_read", rate_gbytes_s=10, clients=1620)], reserve_fraction=0.25))
        self.assertAlmostEqual(r["directions"]["read"]["sides"]["A"]["ns_aggregate_uplink"]["capacity_tbps"], 252.0)


class Directionality(unittest.TestCase):
    def test_read_and_write_not_summed(self):
        # 3.6 Tb/s each way on 4 Tb/s per side per direction: within, though the sum (7.2) would exceed
        flows = [sm.Flow("r", "training_read", rate_gbytes_s=450, clients=1620),
                 sm.Flow("w", "checkpoint_write", rate_gbytes_s=450, clients=1620)]
        r = sm.evaluate(study(flows, side_split={"A": 1.0, "B": 0.0}))
        for d in ("read", "write"):
            self.assertAlmostEqual(r["directions"][d]["sides"]["A"]["storage_attachment"]["utilisation"], 0.9)

    def test_collective_rejected(self):
        with self.assertRaises(ValueError):
            sm.Flow("ar", "all_reduce", rate_gbytes_s=1)

    def test_cache_hit_on_write_rejected(self):
        with self.assertRaises(ValueError):
            sm.Flow("w", "checkpoint_write", unique_tb=1, deadline_s=1, cache_hit=0.5, clients=1)


class CacheFanout(unittest.TestCase):
    def test_full_cache_hit_zero_network(self):
        f = sm.Flow("r", "training_read", unique_tb=100, deadline_s=10, cache_hit=1.0, clients=10)
        self.assertEqual(f.payload_tbps(), 0.0)

    def test_fanout_multiplies_wire_bytes(self):
        f1 = sm.Flow("a", "restart_read", unique_tb=50, fanout=1, deadline_s=300, clients=1620)
        f8 = sm.Flow("b", "restart_read", unique_tb=50, fanout=8, deadline_s=300, clients=1620)
        self.assertAlmostEqual(f8.payload_tbps() / f1.payload_tbps(), 8.0)
        self.assertAlmostEqual(f8.wire_tb(), 400.0)

    def test_cache_and_fanout_combined(self):
        f = sm.Flow("s", "dataset_staging", unique_tb=10, fanout=1620, cache_hit=0.5, deadline_s=14400, clients=1620)
        self.assertAlmostEqual(f.wire_tb(), 8100.0)


class RootF01Regression(unittest.TestCase):
    """review/ROOT_F01_MODEL_REPRO.json: 100 GB/s read to 1 client + 1 GB/s write from 1,620 clients."""

    def flows(self, write=True):
        f = [sm.Flow("hot", "restart_read", rate_gbytes_s=100.0, clients=1, group="hot")]
        if write:
            f.append(sm.Flow("bulk", "checkpoint_write", rate_gbytes_s=1.0, clients=1620, group="all"))
        return f

    def test_hot_read_client_not_diluted(self):
        r = sm.evaluate(study(self.flows(), payload_efficiency=0.9, reserve_fraction=0.1))
        c = rcut(r)
        self.assertAlmostEqual(c["demand_tbps"], 0.4444444444444445)
        self.assertAlmostEqual(c["capacity_tbps"], 0.36)
        self.assertAlmostEqual(c["utilisation"], 1.2345679012345678)
        self.assertEqual(c["verdict"], "exceeds")
        self.assertTrue(any("client_port exceeds" in f for f in r["flags"]))

    def test_opposite_direction_flow_has_no_effect(self):
        a = sm.evaluate(study(self.flows(True), payload_efficiency=0.9, reserve_fraction=0.1))
        b = sm.evaluate(study(self.flows(False), payload_efficiency=0.9, reserve_fraction=0.1))
        self.assertAlmostEqual(rcut(a)["utilisation"], rcut(b)["utilisation"])
        self.assertAlmostEqual(rcut(a, "write")["demand_tbps"], 1.0 * 0.008 / 0.9 * 0.5 / 1620)

    def test_negative_rate_rejected(self):
        with self.assertRaises(ValueError):
            sm.Flow("neg", "restart_read", rate_gbytes_s=-100.0, clients=1)


class ClientGroups(unittest.TestCase):
    def two(self, rate, group2="g2", clients2=1, **kw):
        f = [sm.Flow("a", "restart_read", rate_gbytes_s=rate, clients=1, group="g1"),
             sm.Flow("b", "training_read", rate_gbytes_s=rate, clients=clients2, group=group2)]
        return sm.evaluate(study(f, **kw))

    def test_same_group_coincident_load_adds(self):
        r = self.two(30.0, group2="g1")              # 2 x 0.24 Tb/s x 0.5 = 0.24 Tb/s per side on one tray
        self.assertAlmostEqual(rcut(r)["demand_tbps"], 0.24)
        self.assertEqual(rcut(r)["basis"], "exact")
        self.assertEqual(rcut(self.two(60.0, group2="g1"))["verdict"], "exceeds")

    def test_unknown_relation_envelope_hold(self):
        self.assertEqual(rcut(self.two(30.0))["verdict"], "within")   # envelope 0.24 <= 0.4
        r = self.two(60.0)                                              # each 0.24, envelope 0.48 > 0.4
        c = rcut(r)
        self.assertEqual(c["verdict"], "HOLD")
        self.assertAlmostEqual(c["demand_tbps"], 0.48)
        self.assertAlmostEqual(c["demand_lower_tbps"], 0.24)
        self.assertTrue(c["basis"].startswith("envelope"))
        self.assertIsNone(r["directions"]["read"]["deadline_stretch"])
        self.assertEqual(rcut(self.two(120.0))["verdict"], "exceeds")  # each 0.48 > 0.4 whatever the overlap

    def test_disjoint_relation_exact(self):
        r = self.two(60.0, group_relation="disjoint")
        self.assertEqual(rcut(r)["verdict"], "within")
        self.assertAlmostEqual(rcut(r)["demand_tbps"], 0.24)

    def test_group_client_count_must_match(self):
        with self.assertRaises(ValueError):
            self.two(10.0, group2="g1", clients2=5)

    def test_disjoint_groups_cannot_exceed_trays(self):
        f = [sm.Flow("a", "restart_read", rate_gbytes_s=1, clients=1000, group="g1"),
             sm.Flow("b", "restart_read", rate_gbytes_s=1, clients=1000, group="g2")]
        with self.assertRaises(ValueError):
            sm.evaluate(study(f, group_relation="disjoint"))

    def test_leaf_envelope_concentrated_group(self):
        # 54 trays at 10 GB/s each: 4.32 Tb/s, 2.16 per side, all on one leaf in the worst placement
        f = [sm.Flow("g", "restart_read", rate_gbytes_s=540, clients=54, group="g")]
        c = rcut(sm.evaluate(study(f)), n="ns_leaf_uplink")
        self.assertAlmostEqual(c["demand_tbps"], 2.16)
        self.assertAlmostEqual(c["demand_lower_tbps"], 2.16 / 30)

    def test_leaf_envelope_hold_between_bounds(self):
        # 54 trays at 75 GB/s: 0.3 Tb/s per tray per side; worst leaf 16.2 > 11.2, average 0.54 < 11.2
        f = [sm.Flow("g", "restart_read", rate_gbytes_s=54 * 75, clients=54, group="g")]
        c = rcut(sm.evaluate(study(f, storage_ports={"A": 40, "B": 40})), n="ns_leaf_uplink")
        self.assertAlmostEqual(c["demand_tbps"], 16.2)
        self.assertEqual(c["verdict"], "HOLD")


class SA3Physical(unittest.TestCase):
    def sc(self, R, W, ports=46, H=30, **kw):
        flows = [sm.Flow("r", "training_read", rate_gbytes_s=R, clients=1620, group="all"),
                 sm.Flow("w", "checkpoint_write", rate_gbytes_s=W, clients=1620, group="all")]
        p = dict(attachment="leaf_distributed", leaf_storage_ports=ports, sa3_hosting_leaves=H, sa3_placement="uniform")
        p.update(kw)
        return study(flows, **p)

    def test_read_and_write_share_physical_direction(self):
        # 400 Tb/s line each way (200 per side): each alone 193.3 < 336, together 386.7 > 336
        r = sm.evaluate(self.sc(50000, 50000))
        pc = r["physical_cuts"]["A"]
        self.assertAlmostEqual(pc["ns_aggregate_up"]["demand_tbps"], 400 * 29 / 30)
        self.assertEqual(pc["ns_aggregate_up"]["verdict"], "exceeds")
        self.assertAlmostEqual(pc["ns_leaf_up"]["demand_tbps"], 200 / 30 + 200 / 1620 * 54)
        self.assertTrue(any(f.startswith("physical/A: ns_aggregate_up exceeds") for f in r["flags"]))
        self.assertLess(200 * 29 / 30, 336)          # a logical read-only check would have passed

    def test_placement_required(self):
        r = sm.evaluate(self.sc(100, 100, sa3_placement=None))
        self.assertEqual(r["result_state"], "NOT_EVALUATED_HOLD")
        self.assertIn("sa3_placement", r["missing_inputs"])

    def test_failure_states_hold(self):
        for fl in ("side_lost:A", "ns_leaf_lost:A"):
            r = sm.evaluate(self.sc(100, 100, failure=fl, redirect=1.0))
            self.assertEqual(r["result_state"], "NOT_EVALUATED_HOLD")
            self.assertIn("sa3_failure_replica_placement", r["missing_inputs"])

    def test_per_hosting_leaf_bottleneck(self):
        # 10 hosting leaves, 2 ports each: per hosting leaf read = 14/10 = 1.4 Tb/s > 0.8
        r = sm.evaluate(self.sc(3150, 1575, ports=2, H=10, payload_efficiency=0.9))
        c = rcut(r, n="storage_attachment")
        self.assertAlmostEqual(c["demand_tbps"], 1.4)
        self.assertEqual(c["verdict"], "exceeds")
        self.assertEqual(rcut(r)["verdict"], "within")
        self.assertEqual(r["directions"]["read"]["sides"]["A"]["storage_ports_required"], 4)


class InputDomain(unittest.TestCase):
    def test_nonfinite_and_negative_flow_values(self):
        for kw in ({"rate_gbytes_s": float("nan")}, {"rate_gbytes_s": float("inf")}, {"rate_gbytes_s": -1},
                   {"unique_tb": -1, "deadline_s": 10}, {"unique_tb": 1, "deadline_s": 0},
                   {"unique_tb": 1, "deadline_s": -5}, {"unique_tb": 1, "deadline_s": float("nan")},
                   {"unique_tb": float("inf"), "deadline_s": 10}):
            with self.assertRaises(ValueError, msg=str(kw)):
                sm.Flow("x", "restart_read", clients=10, **kw)
        for kw in ({"fanout": float("nan")}, {"cache_hit": -0.1}, {"cache_hit": float("nan")},
                   {"clients": 0}, {"clients": 1621}, {"clients": 1.5}):
            with self.assertRaises(ValueError, msg=str(kw)):
                sm.Flow("x", "restart_read", rate_gbytes_s=1, **{"clients": 10, **kw})

    def test_zero_rate_is_valid(self):
        r = sm.evaluate(study([sm.Flow("z", "restart_read", rate_gbytes_s=0.0, clients=10)]))
        self.assertEqual(rcut(r)["verdict"], "within")

    def test_side_split_domain(self):
        f = [sm.Flow("r", "training_read", rate_gbytes_s=100, clients=1620)]
        for split in ({"A": 1.5, "B": -0.5}, {"A": float("nan"), "B": 0.5}, {"A": 0.6, "B": 0.6}, {"A": 1.0}):
            with self.assertRaises(ValueError, msg=str(split)):
                sm.evaluate(study(f, side_split=split))
        sm.evaluate(study(f, side_split={"A": 1.0, "B": 0.0}))

    def test_scenario_scalars(self):
        f = [sm.Flow("r", "training_read", rate_gbytes_s=100, clients=1620)]
        for kw in ({"payload_efficiency": 0}, {"payload_efficiency": 1.1}, {"reserve_fraction": 1.0},
                   {"reserve_fraction": -0.1}, {"redirect": 1.5, "failure": "side_lost:A"},
                   {"clients_per_leaf_max": 55}, {"border_reserved_ports": 57},
                   {"failure": "side_lost:C", "redirect": 1}, {"failure": "normal:A"}, {"status": "FACT"}):
            with self.assertRaises(ValueError, msg=str(kw)):
                sm.evaluate(study(f, **kw))

    def test_cvs_pool_boundary_and_explicit_delta(self):
        f = [sm.Flow("r", "training_read", rate_gbytes_s=100, clients=1620)]
        sm.evaluate(study(f, storage_ports={"A": 40, "B": 40}, border_reserved_ports=16))
        with self.assertRaises(ValueError):
            sm.evaluate(study(f, storage_ports={"A": 41, "B": 40}, border_reserved_ports=16))
        r = sm.evaluate(study(f, storage_ports={"A": 70, "B": 70}, border_reserved_ports=16,
                              topology_delta="STUDY: added stage with uplinks outside the CVS pool (unselected)"))
        self.assertTrue(any("unselected topology delta" in x for x in r["flags"]))
        with self.assertRaises(ValueError):
            sm.evaluate(study(f, storage_ports={"A": 70, "B": 70}, topology_delta="  "))

    def test_ns_leaf_port_boundary(self):
        f = [sm.Flow("r", "training_read", rate_gbytes_s=100, clients=1620)]
        base = dict(attachment="leaf_distributed", sa3_hosting_leaves=30, sa3_placement="uniform")
        sm.evaluate(study(f, leaf_storage_ports=46, **base))
        with self.assertRaises(ValueError):
            sm.evaluate(study(f, leaf_storage_ports=47, **base))
        with self.assertRaises(ValueError):
            sm.evaluate(study(f, leaf_storage_ports=2, **dict(base, sa3_hosting_leaves=31)))

    def test_hold_scenario_still_validated(self):
        f = [sm.Flow("r", "training_read")]
        with self.assertRaises(ValueError):
            sm.evaluate(study(f, status="HOLD", side_split={"A": 2.0, "B": -1.0}))

    def test_required_ports_domain(self):
        with self.assertRaises(ValueError):
            sm.required_ports(-1.0)
        self.assertEqual(sm.required_ports(14.0), 35)


class Degraded(unittest.TestCase):
    base_flows = [sm.Flow("r", "training_read", rate_gbytes_s=1000, clients=1620)]

    def test_side_loss_full_redirect_doubles_surviving_demand(self):
        n = sm.evaluate(study(self.base_flows))
        d = sm.evaluate(study(self.base_flows, failure="side_lost:A", redirect=1.0))
        self.assertAlmostEqual(d["directions"]["read"]["sides"]["B"]["storage_attachment"]["demand_tbps"],
                               2 * n["directions"]["read"]["sides"]["B"]["storage_attachment"]["demand_tbps"])
        # surviving capacity is not halved: side B keeps its own full capacity
        self.assertAlmostEqual(d["directions"]["read"]["sides"]["B"]["storage_attachment"]["capacity_tbps"], 4.0)
        self.assertNotIn("A", d["directions"]["read"]["sides"])

    def test_side_loss_without_redirect_reports_unserved(self):
        d = sm.evaluate(study(self.base_flows, failure="side_lost:A", redirect=0.0))
        self.assertAlmostEqual(d["directions"]["read"]["unserved_line_tbps"], 4.0)
        self.assertAlmostEqual(d["directions"]["read"]["sides"]["B"]["storage_attachment"]["utilisation"], 1.0)
        self.assertIsNone(d["directions"]["read"]["deadline_stretch"])

    def test_partial_redirect_flags_and_no_stretch(self):
        d = sm.evaluate(study(self.base_flows, failure="side_lost:A", redirect=0.5))
        v = d["directions"]["read"]
        self.assertAlmostEqual(v["unserved_line_tbps"], 2.0)
        self.assertAlmostEqual(v["served_line_tbps"], 6.0)
        self.assertTrue(v["conservation_ok"])
        self.assertIsNone(v["deadline_stretch"])
        self.assertIn("no path", v["deadline_note"])
        self.assertTrue(any("without a path (redirect=0.5)" in f for f in d["flags"]))

    def test_redirect_required_for_failure(self):
        r = sm.evaluate(study(self.base_flows, failure="side_lost:A"))
        self.assertEqual(r["result_state"], "NOT_EVALUATED_HOLD")
        self.assertIn("redirect", r["missing_inputs"])

    def test_cvs_loss_reduces_uplinks_and_ports(self):
        d = sm.evaluate(study(self.base_flows, failure="cvs_lost:A", cvs_loss_assumption="even_spread_rehome",
                              storage_ports={"A": 14, "B": 14}))
        a = d["directions"]["read"]["sides"]["A"]
        self.assertAlmostEqual(a["ns_aggregate_uplink"]["capacity_tbps"], 288.0)
        self.assertEqual(a["storage_ports_assigned"], 12)
        self.assertAlmostEqual(a["ns_leaf_uplink"]["capacity_tbps"], 9.6)

    def test_cvs_loss_requires_assumption(self):
        r = sm.evaluate(study(self.base_flows, failure="cvs_lost:A"))
        self.assertEqual(r["result_state"], "NOT_EVALUATED_HOLD")
        self.assertIn("cvs_loss_assumption", r["missing_inputs"])
        with self.assertRaises(ValueError):
            sm.evaluate(study(self.base_flows, failure="cvs_lost:A", redirect=1.0, cvs_loss_assumption="even_spread_rehome"))

    def test_cvs_loss_no_rehome_unserved(self):
        d = sm.evaluate(study(self.base_flows, failure="cvs_lost:A", cvs_loss_assumption="even_spread_no_rehome",
                              storage_ports={"A": 14, "B": 14}))
        v = d["directions"]["read"]
        self.assertAlmostEqual(v["unserved_line_tbps"], 4.0 * 2 / 14)
        self.assertTrue(v["conservation_ok"])
        self.assertIsNone(v["deadline_stretch"])

    def test_demand_conserved_in_all_failure_modes(self):
        for kw in ({"failure": "side_lost:B", "redirect": 0.3}, {"failure": "ns_leaf_lost:A", "redirect": 0.0},
                   {"failure": "ns_leaf_lost:B", "redirect": 1.0},
                   {"failure": "cvs_lost:B", "cvs_loss_assumption": "even_spread_no_rehome"}):
            v = sm.evaluate(study(self.base_flows, **kw))["directions"]["read"]
            self.assertAlmostEqual(v["served_line_tbps"] + v["unserved_line_tbps"], v["line_tbps"], msg=str(kw))
            self.assertTrue(v["conservation_ok"])

    def test_ns_leaf_loss_moves_only_affected_trays(self):
        d = sm.evaluate(study(self.base_flows, failure="ns_leaf_lost:A", redirect=1.0))
        a = d["directions"]["read"]["sides"]["A"]["storage_attachment"]["demand_tbps"]
        b = d["directions"]["read"]["sides"]["B"]["storage_attachment"]["demand_tbps"]
        self.assertAlmostEqual(a + b, 8.0)
        self.assertAlmostEqual(b - a, 8.0 * 54 / 1620)


class NSLeafLossBounds(unittest.TestCase):
    """run/grok_02.md B-02: one failed NS leaf holds 54 tray slots in total, not 54 per client group."""

    def grok(self, relation, redirect=1.0):
        f = [sm.Flow("g1", "restart_read", rate_gbytes_s=1250, clients=100, group="g1"),
             sm.Flow("g2", "restart_read", rate_gbytes_s=1250, clients=100, group="g2")]
        return sm.evaluate(study(f, storage_ports={"A": 12, "B": 40}, group_relation=relation,
                                 failure="ns_leaf_lost:A", redirect=redirect))

    def test_disjoint_groups_impaired_side_exceeds(self):
        r = self.grok("disjoint")
        v = r["directions"]["read"]
        a, b = v["sides"]["A"]["storage_attachment"], v["sides"]["B"]["storage_attachment"]
        self.assertAlmostEqual(a["demand_lower_tbps"], 7.3)
        self.assertAlmostEqual(a["demand_tbps"], 10.0)
        self.assertAlmostEqual(a["capacity_tbps"], 4.8)
        self.assertEqual(a["verdict"], "exceeds")
        self.assertAlmostEqual(b["demand_lower_tbps"], 10.0)
        self.assertAlmostEqual(b["demand_tbps"], 12.7)
        self.assertEqual(b["verdict"], "within")
        self.assertAlmostEqual(v["deadline_stretch"], 10.0 / 4.8)
        self.assertAlmostEqual(v["deadline_stretch_lower"], 7.3 / 4.8)
        self.assertTrue(v["conservation_ok"])
        self.assertTrue(any(x.startswith("read/A: storage_attachment exceeds") for x in r["flags"]))

    def test_unknown_overlap_hold_not_within(self):
        v = self.grok("unknown")["directions"]["read"]
        a, b = v["sides"]["A"]["storage_attachment"], v["sides"]["B"]["storage_attachment"]
        self.assertAlmostEqual(a["demand_lower_tbps"], 4.6)
        self.assertAlmostEqual(a["demand_tbps"], 10.0)
        self.assertEqual(a["verdict"], "HOLD")
        self.assertAlmostEqual(b["demand_tbps"], 15.4)
        self.assertEqual(b["verdict"], "within")
        self.assertIsNone(v["deadline_stretch"])
        self.assertIn("HOLD", v["deadline_note"])

    def test_partial_redirect_unserved_is_a_range(self):
        v = self.grok("disjoint", redirect=0.5)["directions"]["read"]
        self.assertAlmostEqual(v["unserved_line_lower_tbps"], 0.0)
        self.assertAlmostEqual(v["unserved_line_tbps"], 2.7 * 0.5)
        self.assertAlmostEqual(v["served_line_tbps"] + v["unserved_line_tbps"], 20.0)
        self.assertAlmostEqual(v["served_line_upper_tbps"], 20.0)
        self.assertTrue(v["conservation_ok"])
        self.assertIsNone(v["deadline_stretch"])
        self.assertIn("may have no path", v["deadline_note"])

    def test_forced_trays_set_lower_bound(self):
        # 1,600 trays cannot all fit on the other 29 leaves (1,566 slots): 34 are forced behind the failed leaf
        f = [sm.Flow("big", "restart_read", rate_gbytes_s=1600 * 1.25, clients=1600, group="big"),
             sm.Flow("small", "restart_read", rate_gbytes_s=100 * 1.25, clients=100, group="small")]
        v = sm.evaluate(study(f, storage_ports={"A": 40, "B": 40}, failure="ns_leaf_lost:A",
                              redirect=1.0))["directions"]["read"]
        side = (1600 + 100) * 0.01 * 0.5             # 0.01 Tb/s per tray, half per side
        a = v["sides"]["A"]["storage_attachment"]
        self.assertAlmostEqual(a["demand_tbps"], side - 34 * 0.005)
        self.assertAlmostEqual(a["demand_lower_tbps"], side - 54 * 0.005 * 2)
        # the large group is certainly affected, so its trays carry the redirected share in the lower bound
        self.assertAlmostEqual(v["sides"]["B"]["client_port"]["demand_lower_tbps"], 0.01)

    def test_disjoint_forced_lightest_trays(self):
        f = [sm.Flow("h", "restart_read", rate_gbytes_s=1000 * 2.5, clients=1000, group="h"),
             sm.Flow("l", "restart_read", rate_gbytes_s=600 * 1.25, clients=600, group="l")]
        v = sm.evaluate(study(f, storage_ports={"A": 40, "B": 40}, group_relation="disjoint",
                              failure="ns_leaf_lost:A", redirect=1.0))["directions"]["read"]
        side = 1000 * 0.01 + 600 * 0.005
        a = v["sides"]["A"]["storage_attachment"]
        self.assertAlmostEqual(a["demand_tbps"], side - 34 * 0.005)      # lightest forced trays
        self.assertAlmostEqual(a["demand_lower_tbps"], side - 54 * 0.01)  # heaviest 54 trays lost

    def test_published_s12_single_group_unchanged(self):
        r = {x.id: sm.evaluate(x) for x in scenarios.build()}["S12"]
        v = r["directions"]["read"]
        b = v["sides"]["B"]
        self.assertEqual(b["storage_ports_required"], 16)
        self.assertAlmostEqual(b["storage_attachment"]["utilisation"], 0.95679012345679)
        self.assertEqual(b["storage_attachment"]["basis"], "exact")
        self.assertEqual(v["sides"]["A"]["storage_attachment"]["basis"], "exact")
        self.assertEqual(b["client_port"]["basis"], "exact")
        self.assertIsNotNone(v["deadline_stretch"])

    def test_no_blanket_hold(self):
        holds = set()
        for s in scenarios.build():
            r = sm.evaluate(s)
            for v in r["directions"].values():
                for side in v["sides"].values():
                    if any(side[n]["verdict"] == "HOLD" for n in ("client_port", "ns_leaf_uplink",
                                                                  "ns_aggregate_uplink", "storage_attachment")):
                        holds.add(s.id)
        self.assertEqual(holds, {"S14"})
        n = sm.evaluate(study([sm.Flow("g", "restart_read", rate_gbytes_s=1000, clients=1620)],
                              failure="ns_leaf_lost:B", redirect=1.0))
        self.assertIsNotNone(n["directions"]["read"]["deadline_stretch"])

    def test_pairwise_pigeonhole_lower_bound(self):
        # two groups of 1,000 trays on 1,620 trays share at least 380 trays: 0.25 + 0.25 = 0.5 > 0.4
        f = [sm.Flow("a", "restart_read", rate_gbytes_s=1000 * 62.5, clients=1000, group="a"),
             sm.Flow("b", "restart_read", rate_gbytes_s=1000 * 62.5, clients=1000, group="b")]
        c = rcut(sm.evaluate(study(f, storage_ports={"A": 40, "B": 40}, border_reserved_ports=16)))
        self.assertAlmostEqual(c["demand_lower_tbps"], 0.5)
        self.assertEqual(c["verdict"], "exceeds")

    def test_groups_must_fit_declared_leaf_capacity(self):
        with self.assertRaises(ValueError):
            sm.evaluate(study([sm.Flow("g", "restart_read", rate_gbytes_s=1, clients=1620)], clients_per_leaf_max=50))


class Figure22Data(unittest.TestCase):
    """run/grok_02.md B-01: each bar pair takes every value from one side and one direction."""

    def test_exact_tuples(self):
        import ports_chart as fg
        rows = {x["id"]: x for x in fg.ports_chart_data([sm.evaluate(s) for s in scenarios.build()])}
        t = lambda i: tuple(rows[i][k] for k in ("side", "direction", "required", "nopath_ports", "nopath_kind", "assigned", "pool"))
        self.assertEqual(t("S06"), ("A", "read", 35, 0, "", 30, 34))
        self.assertEqual(t("S06n"), ("A", "read", 30, 6, "impaired_side", 30, 34))
        self.assertEqual(t("S05"), ("B", "read", 35, 35, "lost_side", 36, 40))
        self.assertEqual(t("S04"), ("B", "read", 70, 0, "", 36, 40))
        self.assertEqual(t("S12"), ("B", "read", 16, 0, "", 16, 40))
        self.assertNotIn("S11", rows)


class HoldGate(unittest.TestCase):
    def test_hold_scenario_not_evaluated(self):
        p = [s for s in scenarios.build() if s.id == "P00"][0]
        r = sm.evaluate(p)
        self.assertEqual(r["result_state"], "NOT_EVALUATED_HOLD")
        self.assertEqual(r["directions"], {})
        self.assertNotIn("within", str(r["flags"]))

    def test_storage_endpoint_limit_hold(self):
        r = sm.evaluate(study([sm.Flow("r", "training_read", rate_gbytes_s=100, clients=10)]))
        self.assertEqual(r["directions"]["read"]["storage_endpoint"]["verdict"], "HOLD")

    def test_study_scenarios_evaluate(self):
        for s in scenarios.build():
            r = sm.evaluate(s)
            self.assertIn(r["result_state"], ("EVALUATED_STUDY", "NOT_EVALUATED_HOLD"))
            for v in r["directions"].values():
                self.assertTrue(v["conservation_ok"], s.id)

    def test_ra_scaling(self):
        self.assertAlmostEqual(scenarios.ra_scaled("enhanced_read"), 3150.0)


