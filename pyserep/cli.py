#!/usr/bin/env python3
"""
pyserep  —  Command-line interface for the SEREP ROM pipeline.

Usage examples
--------------
Full range:
  pyserep --stiffness K.mtx --mass M.mtx --force-dofs 3000 --output-dofs 3000

Selective bands + direct FRF:
  pyserep -k K.mtx -m M.mtx -f 3000 -o 3000 \\
      --bands "0,100,Low" "400,500,High" --frf-method direct

Multiple DOF pairs:
  pyserep -k K.mtx -m M.mtx \\
      -f 3000 5000 -o 3000 5000 \\
      --bands "0,100" "400,500"
"""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pyserep",
        description="SEREP Reduced Order Model pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Required
    p.add_argument("--stiffness", "-k", required=True, help="Path to K matrix (.mtx/.npz)")
    p.add_argument("--mass",      "-m", required=True, help="Path to M matrix (.mtx/.npz)")
    p.add_argument("--force-dofs",  "-f", nargs="+", type=int, required=True)
    p.add_argument("--output-dofs", "-o", nargs="+", type=int, required=True)

    # Frequency
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--bands", nargs="+", metavar="BAND",
                     help="'f_min,f_max[,label]' per band")
    grp.add_argument("--freq-range", nargs=2, type=float, metavar=("FMIN","FMAX"),
                     default=[0.1, 500.0])

    # FRF
    p.add_argument("--frf-method",   default="direct", choices=["direct", "modal"])
    p.add_argument("--damping-type", default="modal",
                   choices=["modal", "rayleigh", "hysteretic", "none"])
    p.add_argument("--zeta",         type=float, default=0.001)
    p.add_argument("--points",       type=int,   default=2000, dest="n_points")

    # Eigensolver
    p.add_argument("--num-modes", type=int, default=100)

    # Mode selection
    p.add_argument("--ms1-alpha",     type=float, default=1.5)
    p.add_argument("--ms2-threshold", type=float, default=1.0)
    p.add_argument("--ms3-threshold", type=float, default=5.0)
    p.add_argument("--mac-threshold", type=float, default=0.90)
    p.add_argument("--rb-hz",         type=float, default=1.0)

    # DOF selection
    p.add_argument("--dof-method", default="eid",
                   choices=["eid", "kinetic", "modal_disp", "svd"])

    # Output
    p.add_argument("--output-folder", default="pyserep_output")
    p.add_argument("--prefix",        default="SEREP")
    p.add_argument("--no-plot",       action="store_true")
    p.add_argument("--quiet", "-q",   action="store_true")
    p.add_argument("--version",       action="version", version="pyserep 3.0.0")
    return p


def _parse_band(s: str):
    from pyserep.selection.band_selector import FrequencyBand
    parts = s.strip().split(",")
    if len(parts) < 2:
        raise argparse.ArgumentTypeError(f"Bad band spec '{s}': need f_min,f_max[,label]")
    label = parts[2].strip() if len(parts) > 2 else None
    return FrequencyBand(float(parts[0]), float(parts[1]), label=label)


def main() -> int:
    """Entry point for the ``pyserep`` command-line interface."""
    parser  = _build_parser()
    args    = parser.parse_args()

    from pyserep.pipeline.config import ROMConfig
    from pyserep.pipeline.serep_pipeline import SereпPipeline

    if args.bands:
        try:
            bands = [_parse_band(b) for b in args.bands]
            band_kw = {"bands": bands}
        except Exception as exc:
            print(f"Error parsing bands: {exc}", file=sys.stderr)
            return 1
    else:
        band_kw = {"freq_range": tuple(args.freq_range)}

    try:
        cfg = ROMConfig(
            stiffness_file    = args.stiffness,
            mass_file         = args.mass,
            force_dofs        = args.force_dofs,
            output_dofs       = args.output_dofs,
            frf_method        = args.frf_method,
            damping_type      = args.damping_type,
            zeta              = args.zeta,
            n_points_per_band = args.n_points,
            num_modes_eigsh   = args.num_modes,
            ms1_alpha         = args.ms1_alpha,
            ms2_threshold     = args.ms2_threshold,
            ms3_threshold     = args.ms3_threshold,
            mac_threshold     = args.mac_threshold,
            rb_hz             = args.rb_hz,
            dof_method        = args.dof_method,
            export_folder     = args.output_folder,
            save_prefix       = args.prefix,
            plot              = not args.no_plot,
            verbose           = not args.quiet,
            **band_kw,
        )
        SereпPipeline(cfg).run()
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception:
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
