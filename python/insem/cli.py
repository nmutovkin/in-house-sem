"""Command line: ``insem run config.yaml``, ``insem build config.yaml -o case.h5``,
``insem info case.h5``. Large runs from a case file use the C++ ``insem-run``."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from . import _core, config


def _info(path: str) -> None:
    sim = _core.Simulation.load(path)
    sim.setup()
    m = sim.mesh
    print(f"{path}: {m.n_cells} {m.cell_type} macro-cells ({m.coords}), order {sim.discretization.order}")
    print(f"  elements {sim.dofs.n_elements}, max level {sim.forest.max_level}, "
          f"geometry kinds {sim.geometry_kind_counts()}")
    for region, mat in sim.materials.items():
        print(f"  region {region}: {mat}")
    print(f"  boundaries {sim.boundaries}")
    print(f"  {len(sim.sources)} source(s), {len(sim.receivers)} receiver(s)")
    t = sim.time
    print(f"  time: duration {t.duration}, n_steps {t.n_steps}, dt {t.dt or 'auto'}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="insem", description="insem spectral element simulator")
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="build, run and write outputs of a YAML configuration")
    p_run.add_argument("config")
    p_run.add_argument("-o", "--output", help="output directory (overrides output.directory)")
    p_build = sub.add_parser("build", help="write the case file of a YAML configuration")
    p_build.add_argument("config")
    p_build.add_argument("-o", "--output", default="case.h5", help="case file to write")
    p_info = sub.add_parser("info", help="summarise a case file")
    p_info.add_argument("case")
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            config.run(args.config, args.output)
        elif args.command == "build":
            config.load(args.config).simulation.save(args.output)
            print(f"wrote {args.output}")
        else:
            _info(args.case)
    except (config.ConfigError, _core.InsemError, OSError) as e:
        print(f"insem: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
