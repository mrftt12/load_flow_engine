# lfe-python

`lfe-python` is a three-phase unbalanced load flow engine written in Python. The project mirrors 
typical power system data structures and workflow while exposing a small, scriptable API for building 
distribution networks, assembling the admittance matrix, solving load flow, and exporting results.

The package published by this repository is `load-flow-engine`, and the main Python package is `load_flow_engine`.

## Features

- Three-phase unbalanced load flow using Newton-Raphson or Gauss-Seidel
- Sparse `Y_abc` network assembly for buses, branches, transformers, loads, generators, shunts, and switches
- Per-phase bus voltage and branch loading results
- Built-in 4-bus example feeder
- Backward-compatible root module via `three_phase_loadflow.py`
- Import/export helpers for SQLite, OpenDSS, CIM/CGMES, CYME, and pandapower-oriented conversion utilities
- Diagnostics utilities for common topology, impedance, grounding, and modeling issues

## Installation

Core dependencies are minimal:

```bash
pip install -e .
```

Optional extras defined in `pyproject.toml`:

```bash
pip install -e .[dev]
pip install -e .[notebooks]
```

Some tool modules rely on additional packages that are not currently declared as extras:

- `pandas` for `load_flow_engine.tools.output`
- `pandas` and `pandapower` for `load_flow_engine.tools.multiconductor_adapter`
- `cympy` for `load_flow_engine.tools.cyme_adapter` and a local CYME installation

## Quick Start

```python
from load_flow_engine import build_example_network, ThreePhaseLoadFlowSolver

net = build_example_network()
solver = ThreePhaseLoadFlowSolver(net, method="nr")

converged = solver.solve()
print("Converged:", converged)

solver.print_bus_results()
branch_results = solver.compute_branch_results()
solver.print_branch_results(branch_results)
```

You can also run the bundled example directly:

```bash
lfe-example
python -c "from load_flow_engine import run_example; run_example()"
python three_phase_loadflow.py
```

## Basic Workflow

The typical workflow is:

1. Create a `StudyCase`
2. Build a `Network`
3. Add buses, branches, transformers, and loads/generators
4. Call `solve()`
5. Read solved voltages from each `Bus` or call `compute_branch_results()`

Example skeleton:

```python
import numpy as np

from load_flow_engine import (
    Branch,
    Bus,
    BusType,
    Load,
    Network,
    PhaseType,
    StudyCase,
    ThreePhaseLoadFlowSolver,
)

sc = StudyCase(max_iterations=50, solution_precision=1e-4, base_mva=10.0)
net = Network(sc)

net.add_bus(Bus("SOURCE", BusType.SLACK, PhaseType.ABC, base_kv=12.47))
net.add_bus(Bus("LOADBUS", BusType.PQ, PhaseType.ABC, base_kv=12.47))

net.add_branch(Branch(
    id="LINE1",
    from_bus="SOURCE",
    to_bus="LOADBUS",
    phase_type=PhaseType.ABC,
    r1=0.01,
    x1=0.03,
    r0=0.03,
    x0=0.09,
))

net.add_load(Load(
    id="LD1",
    bus_id="LOADBUS",
    phase_type=PhaseType.ABC,
    mw=np.array([0.2, 0.2, 0.2]),
    mvar=np.array([0.08, 0.08, 0.08]),
))

solver = ThreePhaseLoadFlowSolver(net)
solver.solve()
```

## Public API

The top-level package re-exports the main classes and helpers:

- `StudyCase`
- `Bus`, `Branch`, `Transformer`, `Load`, `Generator`, `Shunt`
- `Network`
- `ThreePhaseLoadFlowSolver`
- `PhaseType` and `BusType`
- `build_example_network()` and `run_example()`

`three_phase_loadflow.py` remains available as a compatibility shim that re-exports the same public symbols.

## Tools And Adapters

- `load_flow_engine.tools.sqlite_adapter`: persist a network to SQLite with `export_network()` and load it back with `import_network()`
- `load_flow_engine.tools.opendss_adapter`: export a populated network to an OpenDSS script
- `load_flow_engine.tools.cim_adapter`: import CIM XML or export CIM/CGMES-style files
- `load_flow_engine.tools.cyme_adapter`: push a network into a CYME study through `cympy`
- `load_flow_engine.tools.diagnostics`: run validation checks with `run_diagnostics()`
- `load_flow_engine.tools.output`: extract bus results into a pandas DataFrame and summarize phase loading
- `load_flow_engine.tools.load_allocation`: allocate total demand across loads based on connected transformer capacity
- `load_flow_engine.tools.multiconductor_adapter`: convert from pandapower-style multiconductor models into an LFE `Network`

## Project Layout

```text
load_flow_engine/
  __init__.py
  models.py
  network.py
  solver.py
  example.py
  tools/
    diagnostics/
    cim_adapter.py
    cyme_adapter.py
    opendss_adapter.py
    sqlite_adapter.py
three_phase_loadflow.py
notebooks/
```

## Notes

- Core runtime dependencies are `numpy` and `scipy`.
- The repository currently does not include an automated test suite.
- The example feeder in this repository solves successfully through the legacy script entrypoint.

## License


