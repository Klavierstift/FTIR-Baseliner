"""Fit and optionally plot an FTIR baseline from an ASP spectrum.

Example:
	python main.py samples/ASP_Sample_FullResolution_Absorbance.csv --plot
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from matplotlib.widgets import Slider, TextBox

from asp_reader import ASP_Reader, detect_signal_type
from asp_writer import ASP_Writer
from tension_fit import fit_baseline, fit_baseline_cloth


def fit_asp_spectrum(
	data,
	rotation_freedom=10.0,
	signal_type="auto",
	baseline_points=226,
	method="cloth",
	stiffness=1e-5,
	fit_ranges=None,
	ignore_ranges=None,
):
	"""Return a baseline for two-column wavenumber/intensity data."""
	data = np.asarray(data, dtype=float)
	if data.ndim != 2 or data.shape[1] != 2 or data.shape[0] < 2:
		raise ValueError("Spectrum data must contain at least two wavenumber/intensity pairs")
	if baseline_points < 2:
		raise ValueError("baseline_points must be at least 2")
	if signal_type == "auto":
		signal_type = detect_signal_type(data)
	if signal_type == "absorbance":
		fit_data = data
		invert_baseline = False
	elif signal_type == "transmittance":
		fit_data = data.copy()
		fit_data[:, 1] *= -1
		invert_baseline = True
	else:
		raise ValueError("signal_type must be 'absorbance' or 'transmittance'")

	if method not in ("cloth", "rotation"):
		raise ValueError("method must be 'cloth' or 'rotation'")
	fit_ranges = _validate_ranges(fit_ranges, "fit_ranges")
	ignore_ranges = _validate_ranges(ignore_ranges, "ignore_ranges")
	fit_mask = np.ones(len(data), dtype=bool)
	if fit_ranges:
		fit_mask[:] = False
		for lower, upper in fit_ranges:
			fit_mask |= (data[:, 0] >= lower) & (data[:, 0] <= upper)
	for lower, upper in ignore_ranges:
		fit_mask &= ~((data[:, 0] >= lower) & (data[:, 0] <= upper))
	selected_data = fit_data[fit_mask]
	if len(selected_data) < 3:
		raise ValueError("At least three spectral points must remain in the fitting regions")

	if baseline_points >= len(selected_data):
		coarse_data = selected_data
	else:
		indices = np.unique(
			np.rint(np.linspace(0, len(selected_data) - 1, baseline_points)).astype(int)
		)
		segment_minima = np.asarray(
			[
				np.min(selected_data[start : end + 1, 1])
				for start, end in zip(indices[:-1], indices[1:])
			],
			dtype=float,
		)
		caps = np.empty(len(indices), dtype=float)
		caps[0] = segment_minima[0]
		caps[-1] = segment_minima[-1]
		caps[1:-1] = np.minimum(segment_minima[:-1], segment_minima[1:])
		coarse_data = np.column_stack((selected_data[indices, 0], caps))

	if method == "cloth":
		coarse_baseline = fit_baseline_cloth(
			coarse_data,
			stiffness=stiffness,
			signal_amplitude=np.ptp(selected_data[:, 1]),
		)
	else:
		coarse_baseline = fit_baseline(
			coarse_data,
			rotation_freedom=rotation_freedom,
			signal_amplitude=np.ptp(selected_data[:, 1]),
		)
	order = np.argsort(coarse_data[:, 0])
	baseline = np.interp(
		data[:, 0], coarse_data[order, 0], coarse_baseline[order]
	)
	if ignore_ranges:
		baseline = _smooth_ignored_bridges(fit_data, baseline, ignore_ranges)

	return -baseline if invert_baseline else baseline


def _validate_ranges(ranges, name):
	validated = []
	for spectral_range in ranges or []:
		if len(spectral_range) != 2:
			raise ValueError(f"Each {name} entry must contain a minimum and maximum wavenumber")
		lower, upper = map(float, spectral_range)
		if not np.isfinite([lower, upper]).all() or lower > upper:
			raise ValueError(f"Each {name} entry must be finite and ordered from low to high")
		validated.append((lower, upper))
	return validated


def _smooth_ignored_bridges(data, baseline, ignored_ranges):
	order = np.argsort(data[:, 0])
	wavenumbers = data[order, 0]
	fitted = baseline[order].copy()
	merged_ranges = []
	for lower, upper in sorted(ignored_ranges):
		if merged_ranges and lower <= merged_ranges[-1][1]:
			merged_ranges[-1] = (merged_ranges[-1][0], max(upper, merged_ranges[-1][1]))
		else:
			merged_ranges.append((lower, upper))

	for lower, upper in merged_ranges:
		if upper < wavenumbers[0] or lower > wavenumbers[-1]:
			continue
		lower = max(lower, float(wavenumbers[0]))
		upper = min(upper, float(wavenumbers[-1]))
		left_index = int(np.searchsorted(wavenumbers, lower, side="left"))
		right_index = int(np.searchsorted(wavenumbers, upper, side="right")) - 1
		if right_index <= left_index:
			continue

		left_x = wavenumbers[left_index]
		right_x = wavenumbers[right_index]
		left_y = fitted[left_index]
		right_y = fitted[right_index]
		secant = (right_y - left_y) / (right_x - left_x)
		left_slope = (
			(left_y - fitted[left_index - 1]) / (left_x - wavenumbers[left_index - 1])
			if left_index > 0
			else secant
		)
		right_slope = (
			(fitted[right_index + 1] - right_y)
			/ (wavenumbers[right_index + 1] - right_x)
			if right_index < len(wavenumbers) - 1
			else secant
		)

		if abs(secant) < 1e-15:
			left_slope = 0.0
			right_slope = 0.0
		else:
			if left_slope * secant <= 0:
				left_slope = 0.0
			if right_slope * secant <= 0:
				right_slope = 0.0
			slope_norm = np.hypot(left_slope / secant, right_slope / secant)
			if slope_norm > 3:
				limiter = 3 / slope_norm
				left_slope *= limiter
				right_slope *= limiter

		bridge = CubicHermiteSpline(
			[left_x, right_x],
			[left_y, right_y],
			[left_slope, right_slope],
		)
		fitted[left_index : right_index + 1] = bridge(
			wavenumbers[left_index : right_index + 1]
		)

	unsorted = np.empty_like(fitted)
	unsorted[order] = fitted
	return unsorted


def show_fit(
	data,
	baseline,
	signal_type,
	fit_ranges=None,
	ignore_ranges=None,
	show_fitted_spectrum=False,
	display_as="source",
	stiffness=None,
	baseline_fitter=None,
):
	if display_as == "source":
		display_as = signal_type
	if display_as not in ("absorbance", "transmittance"):
		raise ValueError("display_as must be 'source', 'absorbance', or 'transmittance'")
	display_data, display_baseline = _convert_for_display(data, baseline, signal_type, display_as)
	if show_fitted_spectrum:
		figure, (axis, fitted_axis) = plt.subplots(2, 1, sharex=True)
	else:
		figure, axis = plt.subplots()
	axis.plot(data[:, 0], display_data, label="Spectrum")
	baseline_line = axis.plot(data[:, 0], display_baseline, label="Baseline fit")[0]
	baseline_corrected_line = None
	if show_fitted_spectrum:
		baseline_corrected = display_data - display_baseline
		baseline_corrected_line = fitted_axis.plot(
			data[:, 0], baseline_corrected, label="Baseline-corrected spectrum"
		)[0]
		fitted_axis.axhline(0, color="black", linewidth=0.8, alpha=0.5)
		fitted_axis.set_ylabel(
			"Baseline-corrected transmittance (%)"
			if display_as == "transmittance"
			else "Baseline-corrected absorbance"
		)
		fitted_axis.set_xlabel("Wavenumber (cm-1)")
	for index, (lower, upper) in enumerate(_validate_ranges(fit_ranges, "fit_ranges")):
		for target_axis in (axis, fitted_axis) if show_fitted_spectrum else (axis,):
			target_axis.axvspan(
				lower,
				upper,
				color="tab:green",
				alpha=0.1,
				label="Fitting region" if index == 0 else None,
			)
	for index, (lower, upper) in enumerate(_validate_ranges(ignore_ranges, "ignore_ranges")):
		for target_axis in (axis, fitted_axis) if show_fitted_spectrum else (axis,):
			target_axis.axvspan(
				lower,
				upper,
				color="tab:red",
				alpha=0.12,
				label="Ignored region" if index == 0 else None,
			)
	axis.set_xlabel("Wavenumber (cm-1)")
	axis.set_ylabel("Transmittance (%)" if display_as == "transmittance" else "Absorbance")
	axis.set_xlim(float(np.max(data[:, 0])), float(np.min(data[:, 0])))
	axis.grid(True, alpha=0.25)
	axis.legend()
	if show_fitted_spectrum:
		fitted_axis.set_xlim(float(np.max(data[:, 0])), float(np.min(data[:, 0])))
		fitted_axis.grid(True, alpha=0.25)
		fitted_axis.legend()
	current_baseline = [np.asarray(baseline, dtype=float)]
	current_stiffness = [stiffness]
	if baseline_fitter is not None and stiffness is not None:
		figure.tight_layout(rect=(0, 0.15, 1, 1))
		stiffness_exponent = np.log10(float(stiffness))
		slider_axis = figure.add_axes((0.18, 0.06, 0.52, 0.04))
		stiffness_slider = Slider(
			slider_axis,
			"Stiffness (log)",
			min(-7.0, stiffness_exponent - 2),
			max(-3.0, stiffness_exponent + 2),
			valinit=stiffness_exponent,
			valstep=0.05,
			valfmt="%1.2f",
		)
		input_axis = figure.add_axes((0.76, 0.055, 0.19, 0.055))
		stiffness_input = TextBox(
			input_axis,
			"Value",
			initial=f"{stiffness:.2g}",
		)
		status_text = figure.text(0.18, 0.015, "", fontsize=9, color="tab:red")

		def update_stiffness(exponent):
			new_stiffness = 10 ** float(exponent)
			try:
				new_baseline = baseline_fitter(new_stiffness)
				_, new_display_baseline = _convert_for_display(
					data, new_baseline, signal_type, display_as
				)
			except (ValueError, RuntimeError) as error:
				status_text.set_text(f"Fit failed: {error}")
				figure.canvas.draw_idle()
				return

			baseline_line.set_ydata(new_display_baseline)
			if baseline_corrected_line is not None:
				baseline_corrected_line.set_ydata(display_data - new_display_baseline)
			current_baseline[0] = np.asarray(new_baseline, dtype=float)
			current_stiffness[0] = new_stiffness
			stiffness_input.set_val(f"{new_stiffness:.2g}")
			status_text.set_text("")
			axis.relim()
			axis.autoscale_view(scalex=False)
			if show_fitted_spectrum:
				fitted_axis.relim()
				fitted_axis.autoscale_view(scalex=False)
			figure.canvas.draw_idle()

		def submit_stiffness(value):
			try:
				new_stiffness = float(value)
				new_exponent = np.log10(new_stiffness)
			except (TypeError, ValueError):
				status_text.set_text("Enter a positive numeric stiffness value.")
				stiffness_input.set_val(f"{current_stiffness[0]:.2g}")
				figure.canvas.draw_idle()
				return
			if not np.isfinite(new_stiffness) or new_stiffness <= 0:
				status_text.set_text("Enter a positive finite stiffness value.")
				stiffness_input.set_val(f"{current_stiffness[0]:.2g}")
				figure.canvas.draw_idle()
				return
			if not stiffness_slider.valmin <= new_exponent <= stiffness_slider.valmax:
				minimum = 10 ** stiffness_slider.valmin
				maximum = 10 ** stiffness_slider.valmax
				status_text.set_text(
					f"Enter a value between {minimum:.2g} and {maximum:.2g}."
				)
				stiffness_input.set_val(f"{current_stiffness[0]:.2g}")
				figure.canvas.draw_idle()
				return
			status_text.set_text("")
			stiffness_slider.set_val(new_exponent)

		stiffness_slider.on_changed(update_stiffness)
		stiffness_input.on_submit(submit_stiffness)
		setattr(figure, "_stiffness_slider", stiffness_slider)
		setattr(figure, "_stiffness_input", stiffness_input)
	else:
		figure.tight_layout()
	plt.show()
	return current_baseline[0], current_stiffness[0]


def _convert_for_display(data, baseline, signal_type, display_as):
	signal = np.asarray(data[:, 1], dtype=float)
	baseline = np.asarray(baseline, dtype=float)
	if signal_type == display_as:
		return signal, baseline
	if signal_type == "transmittance":
		if np.any(signal <= 0) or np.any(baseline <= 0):
			raise ValueError("Transmittance values must be positive to display absorbance")
		with np.errstate(divide="ignore", invalid="ignore"):
			displayed_signal = -np.log10(signal / 100)
			displayed_baseline = -np.log10(baseline / 100)
	else:
		with np.errstate(over="ignore", invalid="ignore"):
			displayed_signal = 100 * np.power(10.0, -signal)
			displayed_baseline = 100 * np.power(10.0, -baseline)
	if not np.isfinite(displayed_signal).all() or not np.isfinite(displayed_baseline).all():
		raise ValueError("Spectrum values cannot be represented in the requested display units")
	return displayed_signal, displayed_baseline


def _choose_input_file():
	try:
		import tkinter as tk
		from tkinter import filedialog
	except ImportError as error:
		raise RuntimeError("A file dialog is unavailable; pass the input path on the command line") from error

	root = tk.Tk()
	root.withdraw()
	try:
		selected_path = filedialog.askopenfilename(
			title="Open FTIR spectrum",
			filetypes=(("ASP and CSV spectra", "*.asp *.csv"), ("All files", "*.*")),
		)
	finally:
		root.destroy()
	return Path(selected_path) if selected_path else None


def main(argv=None):
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"input",
		type=Path,
		nargs="?",
		metavar="INPUT",
		help="ASP or two-column CSV file (opens a file picker when omitted)",
	)
	parser.add_argument(
		"--freedom",
		type=float,
		default=10.0,
		help="Segment rotation freedom in percent (0-100; default: 10)",
	)
	parser.add_argument(
		"--method",
		choices=("cloth", "rotation"),
		default="cloth",
		help="Baseline fitting model (default: cloth)",
	)
	parser.add_argument(
		"--stiffness",
		type=float,
		default=9e-5,
		help="Cloth curvature resistance (default: 9e-5)",
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
		help="Output path for the fitted baseline",
	)
	parser.add_argument(
		"--output-format",
		choices=("csv", "asp"),
		default="asp",
		help="Fitted baseline format (default: asp)",
	)
	parser.add_argument(
		"--plot",
		action="store_true",
		help="Show the spectrum and fitted baseline in a plot",
	)
	parser.add_argument(
		"--show-baseline-corrected",
		"--show-fitted-spectrum",
		dest="show_baseline_corrected",
		action="store_true",
		help="Show the baseline-corrected spectrum in a separate plot panel",
	)
	args = parser.parse_args(argv)
	if args.input is None:
		try:
			args.input = _choose_input_file()
		except (OSError, RuntimeError) as error:
			parser.error(str(error))
		if args.input is None:
			parser.exit(0, "No input file selected.\n")

	try:
		reader = ASP_Reader(args.input)
		data = reader.read()
		signal_type = (
			(args.signal_type if args.signal_type != "auto" else reader.signal_type)
			or detect_signal_type(data)
		)
		baseline = fit_asp_spectrum(
			data,
			args.freedom,
			signal_type,
			args.baseline_points,
			args.method,
			args.stiffness,
			args.fit_range,
			args.ignore_range,
		)
		output_path = args.output or args.input.with_name(
			f"{args.input.stem}_baseline.{args.output_format}"
		)
	except (OSError, ValueError) as error:
		parser.error(str(error))

	if args.plot or args.show_baseline_corrected:
		baseline, _ = show_fit(
			data,
			baseline,
			signal_type,
			args.fit_range,
			args.ignore_range,
			args.show_baseline_corrected,
			args.display_as,
			stiffness=args.stiffness if args.method == "cloth" else None,
			baseline_fitter=(
				lambda selected_stiffness: fit_asp_spectrum(
					data,
					args.freedom,
					signal_type,
					args.baseline_points,
					args.method,
					stiffness=selected_stiffness,
					fit_ranges=args.fit_range,
					ignore_ranges=args.ignore_range,
				)
				if args.method == "cloth"
				else None
			),
		)

	try:
		baseline_data = np.column_stack((data[:, 0], baseline))
		if args.output_format == "asp":
			ASP_Writer(output_path).write(baseline_data)
		else:
			np.savetxt(
				output_path,
				baseline_data,
				delimiter=",",
				header="wavenumber,baseline",
				comments="",
			)
	except (OSError, ValueError) as error:
		parser.error(str(error))

	print(f"Baseline fit saved to {output_path}")


if __name__ == "__main__":
	main()
