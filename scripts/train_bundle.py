"""Entrena la red bundle: aprende δ(a, Ω_m0) para todo un rango de Ω_m0 de una sola vez.

Uso:
    python scripts/train_bundle.py
    python scripts/train_bundle.py --om-m0-min 0.1 --om-m0-max 0.5 --iterations 20000
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from cosmopinn.training import train_bundle  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--om-m0-min", type=float, default=0.1)
    parser.add_argument("--om-m0-max", type=float, default=0.5)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument(
        "--out", type=str, default=None, help="directorio de salida (default: models/bundle_om_010_050)"
    )
    args = parser.parse_args()

    tag = f"bundle_om_{int(round(args.om_m0_min * 100)):03d}_{int(round(args.om_m0_max * 100)):03d}"
    out_dir = Path(args.out) if args.out else ROOT / "models" / tag

    t0 = time.time()
    train_bundle(args.om_m0_min, args.om_m0_max, out_dir, iterations=args.iterations)
    elapsed = time.time() - t0

    print(f"\nEntrenamiento terminado en {elapsed:.1f} s. Run guardado en {out_dir}")


if __name__ == "__main__":
    main()
