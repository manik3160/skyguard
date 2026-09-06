"""Command-line entry point.

    python -m skyguard.cli benchmark [--quick]
    python -m skyguard.cli calibrate
    python -m skyguard.cli spatial-calibrate [--quick]
    python -m skyguard.cli ablation
    python -m skyguard.cli serve [--port 8000] [--speed 12]
    python -m skyguard.cli demo
"""

from __future__ import annotations

import argparse
import json
import sys


def _benchmark(args: argparse.Namespace) -> int:
    from .evaluation.benchmark import run_benchmark
    from .synth.climate import DEMO_NETWORK

    train, test = (2_500, 4_000) if args.quick else (6_000, 12_000)
    report = run_benchmark(DEMO_NETWORK[0], train_samples=train, test_samples=test)
    print(report.render())
    if args.json:
        print()
        print(json.dumps(report.to_dict(), indent=2))
    return 0


def _calibrate(args: argparse.Namespace) -> int:
    from .evaluation.calibrate import calibrate
    from .synth.climate import DEMO_NETWORK

    result = calibrate(DEMO_NETWORK[0], budget=args.budget)
    print(result.render())
    # A failure to meet the false-positive budget is a real result, and the
    # exit code says so, so CI can gate on it.
    return 0 if result.chosen else 1


def _spatial_calibrate(args: argparse.Namespace) -> int:
    from .evaluation.spatial_calibrate import run_spatial_calibration
    from .synth.climate import DEMO_NETWORK

    train, test = (2_000, 2_500) if args.quick else (3_000, 5_000)
    grid = (3.0, 3.5, 4.0) if args.quick else None
    result = run_spatial_calibration(
        DEMO_NETWORK,
        train_samples=train,
        test_samples=test,
        **({"grid": grid} if grid else {}),
    )
    print(result.render())
    # A run where no candidate holds the clean-FPR budget is a real result and
    # the exit code says so, so CI can gate on it.
    return 0 if result.chosen_f1 else 1


def _ablation(args: argparse.Namespace) -> int:
    from .evaluation.network_benchmark import run_network_benchmark
    from .synth.climate import DEMO_NETWORK

    train, test = (2_000, 3_000) if args.quick else (3_000, 6_000)
    report = run_network_benchmark(DEMO_NETWORK, train_samples=train, test_samples=test)
    print(report.render())
    return 0


def _serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed. Run: pip install 'skyguard[api]'", file=sys.stderr)
        return 1

    from .api.server import build_app

    app = build_app(speed=args.speed)
    print(f"SkyGuard console: http://127.0.0.1:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def _demo(args: argparse.Namespace) -> int:
    """Print the problem statement's own example use case, end to end."""
    from .models import Observation
    from .pipeline import SkyGuardPipeline
    from .synth.climate import DEMO_NETWORK, ClimateGenerator

    profile = DEMO_NETWORK[0]
    pipeline = SkyGuardPipeline()
    pipeline.register_station(profile.station_id, profile.altitude_m)
    generator = ClimateGenerator(profile, 1_700_000_000.0, seed=4)
    history = generator.generate(2_000)
    pipeline.fit(history)

    for observation in generator.generate(400):
        pipeline.process(observation)

    last = observation
    # The exact scenario from problem statement 26073: a station reporting
    # 55 degC with very high humidity while its neighbours are normal.
    injected = Observation(
        station_id=profile.station_id,
        timestamp=last.timestamp + 600,
        temperature=55.0,
        pressure=last.pressure,
        humidity=96.0,
    )
    verdict = pipeline.process(injected)
    print(json.dumps(verdict.to_json(), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="skyguard", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("benchmark", help="detection accuracy on injected faults")
    p.add_argument("--quick", action="store_true", help="smaller splits, faster")
    p.add_argument("--json", action="store_true", help="also emit machine-readable output")
    p.set_defaults(func=_benchmark)

    p = sub.add_parser("calibrate", help="sweep the alert threshold")
    p.add_argument("--budget", type=float, default=0.01, help="clean-stream FPR budget")
    p.set_defaults(func=_calibrate)

    p = sub.add_parser("ablation", help="measure the spatial layer's contribution")
    p.add_argument("--quick", action="store_true")
    p.set_defaults(func=_ablation)

    p = sub.add_parser(
        "spatial-calibrate", help="sweep the L4 buddy-check threshold on the network"
    )
    p.add_argument("--quick", action="store_true", help="smaller splits, faster")
    p.set_defaults(func=_spatial_calibrate)

    p = sub.add_parser("serve", help="run the live console")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--speed", type=float, default=12.0,
                   help="simulated intervals per real second")
    p.set_defaults(func=_serve)

    p = sub.add_parser("demo", help="run the problem statement's example use case")
    p.set_defaults(func=_demo)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
