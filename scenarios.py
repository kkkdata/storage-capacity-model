"""Illustrative STUDY scenarios and the HOLD project scenario for storage_model.

No value here is a project workload FACT.  RA-scaled rows take NVIDIA DGX SuperPOD
GB300 RA Table 5 (per scalable unit of 8 DGX GB300 systems) and scale linearly to
90 GPU cabinets; the linear scaling and the transfer from a DGX SuperPOD storage
fabric to this converged BF3 north-south design are STUDY assumptions.
"""
from storage_model import Flow, Scenario

RACKS = 90
RA_SU_RACKS = 8
RA = {  # GB/s per SU, NVIDIA DGX SuperPOD GB300 RA Table 5 (FACT for that RA only)
    "standard_read": 90.0, "standard_write": 45.0,
    "enhanced_read": 280.0, "enhanced_write": 140.0,
}


def ra_scaled(key: str) -> float:
    return RA[key] / RA_SU_RACKS * RACKS


STUDY_COMMON = dict(payload_efficiency=0.90, reserve_fraction=0.10, clients_per_leaf_max=54,
                    border_reserved_ports=16)


def _sc(id, title, flows, ports=16, **kw):
    params = dict(STUDY_COMMON)
    params.update(kw)
    return Scenario(id=id, title=title, status="STUDY", flows=flows,
                    storage_ports={"A": ports, "B": ports}, **params)


def ra_flows(level: str, clients: int = 1620):
    return [Flow(f"{level} read", "training_read", rate_gbytes_s=ra_scaled(f"{level}_read"), clients=clients),
            Flow(f"{level} write", "checkpoint_write", rate_gbytes_s=ra_scaled(f"{level}_write"), clients=clients)]


def build() -> list:
    s = []
    s.append(_sc("S01", "SRC-05 Standard scaled to 90 cabinets, concurrent read and write, SA-1, 16 ports per side",
                 ra_flows("standard")))
    s.append(_sc("S02", "SRC-05 Enhanced scaled, SA-1, 16 ports per side", ra_flows("enhanced")))
    s.append(_sc("S03", "SRC-05 Enhanced scaled, SA-1, 36 ports per side", ra_flows("enhanced"), ports=36))
    s.append(_sc("S04", "S03 with side A lost; all clients and storage move to side B (rho = 1)",
                 ra_flows("enhanced"), ports=36, failure="side_lost:A", redirect=1.0))
    s.append(_sc("S05", "S03 with side A lost; no transfer (clients fixed to a side, rho = 0)",
                 ra_flows("enhanced"), ports=36, failure="side_lost:A", redirect=0.0))
    s.append(_sc("S06", "S03 with one side-A CVS lost; sessions redirected within the side",
                 ra_flows("enhanced"), ports=36, failure="cvs_lost:A", cvs_loss_assumption="even_spread_rehome"))
    s.append(_sc("S06n", "S06 without redirection", ra_flows("enhanced"), ports=36,
                 failure="cvs_lost:A", cvs_loss_assumption="even_spread_no_rehome"))
    s.append(_sc("S07a", "10 TB dataset distributed to the cache of every tray (F = 1620), deadline 4 h",
                 [Flow("staging", "dataset_staging", unique_tb=10.0, fanout=1620, deadline_s=4 * 3600, clients=1620)]))
    s.append(_sc("S07b", "10 TB dataset distributed in shards (F = 1), deadline 4 h",
                 [Flow("staging", "dataset_staging", unique_tb=10.0, fanout=1, deadline_s=4 * 3600, clients=1620)]))
    s.append(_sc("S08", "Synchronous checkpoint write of 50 TB, deadline 120 s",
                 [Flow("ckpt", "checkpoint_write", unique_tb=50.0, deadline_s=120, clients=1620)]))
    s.append(_sc("S09", "Recovery read of 50 TB, 8 data-parallel replicas, deadline 300 s",
                 [Flow("restart", "restart_read", unique_tb=50.0, fanout=8, deadline_s=300, clients=1620)]))
    s.append(_sc("S10", "Training read (Enhanced rate, cache hit 0.8) concurrent with the S08 write",
                 [Flow("train", "training_read", rate_gbytes_s=ra_scaled("enhanced_read"), cache_hit=0.8, clients=1620),
                  Flow("ckpt", "checkpoint_write", unique_tb=50.0, deadline_s=120, clients=1620)]))
    s.append(_sc("S11", "Enhanced scaled, SA-3 on unused NS-leaf ports, 2 ports per leaf",
                 ra_flows("enhanced"), attachment="leaf_distributed", leaf_storage_ports=2,
                 sa3_hosting_leaves=30, sa3_placement="uniform"))
    s.append(_sc("S11f", "S11 with side A lost (replica distribution unknown)", ra_flows("enhanced"), attachment="leaf_distributed",
                 leaf_storage_ports=2, sa3_hosting_leaves=30, sa3_placement="uniform",
                 failure="side_lost:A", redirect=1.0))
    s.append(_sc("S12", "S09 with one side-A NS leaf lost; affected trays move to side B (rho = 1)",
                 [Flow("restart", "restart_read", unique_tb=50.0, fanout=8, deadline_s=300, clients=1620)],
                 failure="ns_leaf_lost:A", redirect=1.0))
    s.append(_sc("S13", "One tray reads 100 GB/s while 1,620 trays write 1 GB/s in total",
                 [Flow("hot read", "restart_read", rate_gbytes_s=100.0, clients=1, group="hot"),
                  Flow("bulk write", "checkpoint_write", rate_gbytes_s=1.0, clients=1620, group="all")]))
    s.append(_sc("S14", "Two single-tray reads of 50 GB/s, group overlap unknown",
                 [Flow("read g1", "restart_read", rate_gbytes_s=50.0, clients=1, group="g1"),
                  Flow("read g2", "restart_read", rate_gbytes_s=50.0, clients=1, group="g2")]))
    s.append(_sc("S14d", "S14 with the two groups declared disjoint",
                 [Flow("read g1", "restart_read", rate_gbytes_s=50.0, clients=1, group="g1"),
                  Flow("read g2", "restart_read", rate_gbytes_s=50.0, clients=1, group="g2")],
                 group_relation="disjoint"))
    s.append(Scenario(id="P00", title="Project scenario; all demand inputs HOLD", status="HOLD",
                      flows=[Flow("dataset", "dataset_staging"), Flow("train", "training_read"),
                             Flow("ckpt", "checkpoint_write"), Flow("restart", "restart_read")],
                      note="dataset size, cache capacity/hit, concurrency, checkpoint size/interval/deadline, "
                           "storage system throughput, client multipath and port assignment are HOLD"))
    return s


def ns_leaf_multigroup_regression(relation: str) -> Scenario:
    """Calculator regression (Grok02 B-02), not a project scenario: two read groups of 100 trays each,
    1,250 GB/s per group, one side-A NS leaf lost, rho = 1.  Only one leaf's 54 tray slots are lost."""
    flows = [Flow("read g1", "restart_read", rate_gbytes_s=1250.0, clients=100, group="g1"),
             Flow("read g2", "restart_read", rate_gbytes_s=1250.0, clients=100, group="g2")]
    return Scenario(id=f"B02-{relation}", title="NS leaf loss, two client groups (regression)", status="STUDY",
                    flows=flows, payload_efficiency=1.0, reserve_fraction=0.0, storage_ports={"A": 12, "B": 40},
                    border_reserved_ports=0, clients_per_leaf_max=54, group_relation=relation,
                    failure="ns_leaf_lost:A", redirect=1.0)
