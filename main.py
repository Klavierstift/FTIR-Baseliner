"""Fit and optionally plot an FTIR baseline from an ASP spectrum.

Example:
	python main.py samples/ASP_Sample_FullResolution_Absorbance.csv --plot
"""


import argparse
from pathlib import Path
import sys

from batch import choose_input_files, fit_batch_spectra
# Re-exported so `from main import ...` keeps working.
from fitting import fit_asp_spectrum  # noqa: F401


def main(argv=None):
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"input",
		type=Path,
		nargs="*",
		metavar="INPUT",
		help="One or more ASP or CSV files (opens a multi-select picker when omitted)",
	)
	parser.add_argument(
		"--stiffness",
		type=float,
		default=9e-5,
		help="Cloth curvature resistance (default: 9e-5)",
	)
	parser.add_argument(
		"--solver",
		choices=("newton", "lbfgsb"),
		default="newton",
		help="Optimizer: projected Newton (default) or the older L-BFGS-B",
	)
	parser.add_argument(
		"--baseline-points",
		type=int,
		default=226,
		help="Number of knots used for the baseline fit (default: 226)",
	)
	parser.add_argument(
		"--signal-type",
		choices=("auto", "transmittance", "absorbance"),
		default="auto",
		help="Intensity representation in the input (default: auto-detect)",
	)
	parser.add_argument(
		"--display-as",
		choices=("source", "absorbance", "transmittance"),
		default="source",
		help="Plot as source units, absorbance, or transmittance (default: source)",
	)
	parser.add_argument(
		"--fit-range",
		type=float,
		nargs=2,
		metavar=("MIN", "MAX"),
		action="append",
		help="Use only this wavenumber range for fitting; may be repeated",
	)
	parser.add_argument(
		"--ignore-range",
		type=float,
		nargs=2,
		metavar=("MIN", "MAX"),
		action="append",
		help="Exclude this wavenumber range from fitting; may be repeated",
	)
	parser.add_argument(
		"--output",
		type=Path,
		help="Output path for a single input file; its .asp or .csv extension sets the format",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		help="Directory for batch corrected spectra (defaults to each input's directory)",
	)
	parser.add_argument(
		"--output-format",
		choices=("csv", "asp"),
		default=None,
		help="Corrected spectrum format (default: asp, or the --output extension)",
	)
	parser.add_argument(
		"--output-units",
		choices=("source", "absorbance", "transmittance"),
		default="source",
		help="Units for saved corrected spectra (default: source)",
	)
	parser.add_argument(
		"--plot",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Show the spectrum and fitted baseline (default; use --no-plot to disable)",
	)
	parser.add_argument(
		"--show-baseline-corrected",
		"--show-fitted-spectrum",
		dest="show_baseline_corrected",
		action=argparse.BooleanOptionalAction,
		default=True,
		help=(
			"Show the baseline-corrected spectrum (default; use "
			"--no-show-baseline-corrected to disable)"
		),
	)
	args = parser.parse_args(argv)
	# No paths on the command line: open the file picker.
	if not args.input:
		try:
			args.input = choose_input_files()
		except (OSError, RuntimeError) as error:
			parser.error(str(error))
		if not args.input:
			parser.exit(0, "No input file selected.\n")
	if args.output is not None and len(args.input) != 1:
		parser.error("--output can only be used with one input; use --output-dir for batch runs")
	if args.output is not None and args.output_dir is not None:
		parser.error("--output and --output-dir cannot be used together")
	try:
		output_paths, failures = fit_batch_spectra(
			args.input,
			output_format=args.output_format,
			output_dir=args.output_dir,
			output_path=args.output,
			stiffness=args.stiffness,
			solver=args.solver,
			baseline_points=args.baseline_points,
			signal_type=args.signal_type,
			fit_ranges=args.fit_range,
			ignore_ranges=args.ignore_range,
			plot=args.plot,
			show_baseline_corrected=args.show_baseline_corrected,
			display_as=args.display_as,
			output_units=args.output_units,
		)
	except (OSError, ValueError) as error:
		parser.error(str(error))
	for output_path in output_paths:
		print(f"Baseline-corrected spectrum saved to {output_path}")
	for input_path, error in failures:
		print(f"Failed to process {input_path}: {error}", file=sys.stderr)
	return 1 if failures else 0


if __name__ == "__main__":
	raise SystemExit(main())
