"""
router_eval — offline RouterBench replay harness for Mesh's model=auto router.

MESH-708 Phase 1 (initial setup): replay routing POLICIES against RouterBench's
precomputed inference outcomes to answer, cheaply and with no live calls, how good
the frozen SUPERMODE_BENCHMARKS routing table actually is. Phase 2 (live catalog
runs through Mesh) is intentionally out of scope here.

Run:  python -m router_eval.replay
"""

__all__ = ["data", "policies", "metrics", "replay", "benchmark_table", "routerbench_bridge"]
