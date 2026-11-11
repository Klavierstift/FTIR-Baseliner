"""Batch processing: fit, review, and save multiple spectra."""


from pathlib import Path

from asp_reader import ASP_Reader, detect_signal_type
from fitting import fit_asp_spectrum, validate_ranges
from output import convert_for_display, write_corrected_spectrum


def choose_input_files():
	# Imported lazily; tkinter is only needed when no paths are given.
	try:
		import tkinter as tk
		from tkinter import filedialog
	except ImportError as error:
		raise RuntimeError("A file dialog is unavailable; pass the input path on the command line") from error

	root = tk.Tk()
	root.withdraw()
	try:
		selected_paths = filedialog.askopenfilenames(
			title="Open FTIR spectra",
			filetypes=(("ASP and CSV spectra", "*.asp *.csv"), ("All files", "*.*")),
		)
	finally:
		root.destroy()
	return [Path(path) for path in selected_paths]



def fit_batch_spectra(
	input_paths,
	*,
	output_format=None,
	output_dir=None,
	output_path=None,
	stiffness=9e-5,
	baseline_points=226,
	signal_type="auto",
	fit_ranges=None,
	ignore_ranges=None,
	plot=True,
	show_baseline_corrected=True,
	display_as="source",
	output_units="source",
	solver="newton",
):
	"""Fit baselines and save corrected spectra for multiple ASP or CSV files.

	Returns (output_paths, failures), where failures contains (input_path, error)
	pairs for files that could not be processed.
	"""
	input_paths = [Path(path) for path in input_paths]
	if not input_paths:
		raise ValueError("At least one input file is required")
	if output_format is not None and output_format not in ("asp", "csv"):
		raise ValueError("output_format must be 'asp' or 'csv'")
	if output_path is not None:
		path_format = Path(output_path).suffix.lower().lstrip(".")
		if path_format not in ("asp", "csv"):
			raise ValueError("--output must end in .asp or .csv")
		if output_format is not None and output_format != path_format:
			raise ValueError(
				f"--output ends in .{path_format} but output_format is {output_format}"
			)
		output_format = path_format
	elif output_format is None:
		output_format = "asp"
	if output_units not in ("source", "absorbance", "transmittance"):
		raise ValueError("output_units must be 'source', 'absorbance', or 'transmittance'")
	if output_path is not None and (len(input_paths) != 1 or output_dir is not None):
		raise ValueError("--output can only be used with one input and without --output-dir")
	if output_dir is not None:
		output_dir = Path(output_dir)
		output_dir.mkdir(parents=True, exist_ok=True)
		output_names = [
			f"{path.stem}_corrected.{output_format}".casefold()
			for path in input_paths
		]
		if len(set(output_names)) != len(output_names):
			raise ValueError("Batch inputs have duplicate filenames; use separate input directories")

	items = []
	failures = []
	for input_path in input_paths:
		try:
			reader = ASP_Reader(input_path)
			data = reader.read()
			resolved_signal_type = (
				signal_type if signal_type != "auto" else reader.signal_type
			) or detect_signal_type(data)
			baseline = fit_asp_spectrum(
				data,
				signal_type=resolved_signal_type,
				baseline_points=baseline_points,
				stiffness=stiffness,
				fit_ranges=fit_ranges,
				ignore_ranges=ignore_ranges,
				solver=solver,
			)
			if output_path is not None:
				current_output_path = Path(output_path)
			elif output_dir is not None:
				current_output_path = (
					output_dir / f"{input_path.stem}_corrected.{output_format}"
				)
			else:
				current_output_path = input_path.with_name(
					f"{input_path.stem}_corrected.{output_format}"
				)
			# Check conversions now so bad values fail before the review window opens.
			if plot:
				display_units = resolved_signal_type if display_as == "source" else display_as
				convert_for_display(data, baseline, resolved_signal_type, display_units)
			save_units = resolved_signal_type if output_units == "source" else output_units
			convert_for_display(data, baseline, resolved_signal_type, save_units)
			items.append(
				{
					"input_path": input_path,
					"data": data,
					"signal_type": resolved_signal_type,
					"baseline": baseline,
					"stiffness": stiffness,
					"ignore_ranges": validate_ranges(ignore_ranges, "ignore_ranges"),
					"output_path": current_output_path,
				}
			)
		except (OSError, ValueError, RuntimeError) as error:
			failures.append((input_path, str(error)))

	if plot and items:
		# Imported lazily so headless runs do not load matplotlib.
		from gui import show_batch_fits

		output_paths, review_failures = show_batch_fits(
			items,
			stiffness=stiffness,
			baseline_points=baseline_points,
			fit_ranges=fit_ranges,
			solver=solver,
			show_baseline_corrected=show_baseline_corrected,
			display_as=display_as,
			output_format=output_format,
			output_units=output_units,
		)
		failures.extend(review_failures)
		return output_paths, failures

	output_paths = []
	for item in items:
		baseline = item["baseline"]
		try:
			write_corrected_spectrum(
				item["output_path"],
				item["data"],
				baseline,
				output_format,
				item["signal_type"],
				output_units,
			)
			output_paths.append(item["output_path"])
		except (OSError, ValueError) as error:
			failures.append((item["input_path"], str(error)))
	return output_paths, failures
