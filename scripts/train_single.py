"""Entrena la red no-bundle: aprende δ(a) para un único Ω_m0.

Uso:
    python scripts/train_single.py
    python scripts/train_single.py --om-m0 0.3 --iterations 5000
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from cosmopinn.training import train_single  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--om-m0", type=float, default=0.3, help="Ω_m0 fijo (default: 0.3)")
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--out", type=str, default=None, help="directorio de salida (default: models/single_om030)")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else ROOT / "models" / f"single_om{int(round(args.om_m0 * 100)):03d}"

    t0 = time.time()
    train_single(args.om_m0, out_dir, iterations=args.iterations)
    elapsed = time.time() - t0

    print(f"\nEntrenamiento terminado en {elapsed:.1f} s. Run guardado en {out_dir}")


if __name__ == "__main__":
    main()
