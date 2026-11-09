"""Fit and optionally plot an FTIR baseline from an ASP spectrum.

Example:
	python main.py samples/ASP_Sample_FullResolution_Absorbance.csv --plot
"""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from matplotlib.widgets import Button, Slider, TextBox

from asp_reader import ASP_Reader, detect_signal_type
from asp_writer import ASP_Writer
from tension_fit import fit_baseline_cloth


def fit_asp_spectrum(
	data,
	signal_type="auto",
	baseline_points=226,
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

	coarse_baseline = fit_baseline_cloth(
		coarse_data,
		stiffness=stiffness,
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


def show_batch_fits(
	items,
	*,
	stiffness,
	baseline_points,
	fit_ranges,
	ignore_ranges,
	show_baseline_corrected,
	display_as,
	output_format,
	output_units,
):
	"""Review multiple spectra in one window and save each on confirmation."""
	figure, axes = plt.subplots(2 if show_baseline_corrected else 1, 1, sharex=True)
	if show_baseline_corrected:
		axis, corrected_axis = axes
	else:
		axis = axes
		corrected_axis = None
	figure.tight_layout(rect=(0.05, 0.2, 0.98, 0.91))

	initial_exponent = np.log10(float(stiffness))
	slider_axis = figure.add_axes((0.13, 0.09, 0.43, 0.045))
	stiffness_slider = Slider(
		slider_axis,
		"Stiffness (log)",
		min(-7.0, initial_exponent - 2),
		max(-3.0, initial_exponent + 2),
		valinit=initial_exponent,
		valstep=0.05,
		valfmt="%1.2f",
	)
	input_axis = figure.add_axes((0.61, 0.085, 0.17, 0.055))
	stiffness_input = TextBox(input_axis, "Value", initial=f"{stiffness:.2g}")
	button_axis = figure.add_axes((0.81, 0.085, 0.16, 0.055))
	index = [0]
	saved_paths = []
	failures = []
	status_text = figure.text(0.13, 0.025, "", fontsize=9, color="tab:red")
	save_button = Button(button_axis, "Save / Next")
	updating_controls = [False]

	def draw_current_item():
		item = items[index[0]]
		data = item["data"]
		display_units = item["signal_type"] if display_as == "source" else display_as
		display_data, display_baseline = _convert_for_display(
			data, item["baseline"], item["signal_type"], display_units
		)
		axis.clear()
		axis.plot(data[:, 0], display_data, label="Spectrum")
		axis.plot(data[:, 0], display_baseline, label="Baseline fit")
		axis.set_ylabel("Transmittance (%)" if display_units == "transmittance" else "Absorbance")
		axis.set_xlabel("Wavenumber (cm-1)")
		axis.set_xlim(float(np.max(data[:, 0])), float(np.min(data[:, 0])))
		axis.grid(True, alpha=0.25)
		for range_index, (lower, upper) in enumerate(_validate_ranges(fit_ranges, "fit_ranges")):
			axis.axvspan(
				lower,
				upper,
				color="tab:green",
				alpha=0.1,
				label="Fitting region" if range_index == 0 else None,
			)
		for range_index, (lower, upper) in enumerate(_validate_ranges(ignore_ranges, "ignore_ranges")):
			axis.axvspan(
				lower,
				upper,
				color="tab:red",
				alpha=0.12,
				label="Ignored region" if range_index == 0 else None,
			)
		axis.legend()
		if corrected_axis is not None:
			corrected_axis.clear()
			corrected_axis.plot(
				data[:, 0],
				display_data - display_baseline,
				label="Baseline-corrected spectrum",
			)
			corrected_axis.axhline(0, color="black", linewidth=0.8, alpha=0.5)
			corrected_axis.set_ylabel(
				"Baseline-corrected transmittance (%)"
				if display_units == "transmittance"
				else "Baseline-corrected absorbance"
			)
			corrected_axis.set_xlabel("Wavenumber (cm-1)")
			corrected_axis.set_xlim(float(np.max(data[:, 0])), float(np.min(data[:, 0])))
			corrected_axis.grid(True, alpha=0.25)
			for range_index, (lower, upper) in enumerate(_validate_ranges(fit_ranges, "fit_ranges")):
				corrected_axis.axvspan(
					lower,
					upper,
					color="tab:green",
					alpha=0.1,
					label="Fitting region" if range_index == 0 else None,
				)
			for range_index, (lower, upper) in enumerate(_validate_ranges(ignore_ranges, "ignore_ranges")):
				corrected_axis.axvspan(
					lower,
					upper,
					color="tab:red",
					alpha=0.12,
					label="Ignored region" if range_index == 0 else None,
				)
			corrected_axis.legend()
		figure.suptitle(f"Spectrum {index[0] + 1} of {len(items)}: {item['input_path'].name}")
		updating_controls[0] = True
		stiffness_slider.set_val(np.log10(item["stiffness"]))
		stiffness_input.set_val(f"{item['stiffness']:.2g}")
		updating_controls[0] = False
		save_button.label.set_text(
			"Save / Done" if index[0] == len(items) - 1 else "Save / Next"
		)
		figure.canvas.draw_idle()

	def update_current_stiffness(new_stiffness):
		item = items[index[0]]
		try:
			new_baseline = fit_asp_spectrum(
				item["data"],
				signal_type=item["signal_type"],
				baseline_points=baseline_points,
				stiffness=new_stiffness,
				fit_ranges=fit_ranges,
				ignore_ranges=ignore_ranges,
			)
			display_units = item["signal_type"] if display_as == "source" else display_as
			_convert_for_display(item["data"], new_baseline, item["signal_type"], display_units)
		except (ValueError, RuntimeError) as error:
			status_text.set_text(f"Fit failed: {error}")
			figure.canvas.draw_idle()
			return
		item["baseline"] = new_baseline
		item["stiffness"] = new_stiffness
		status_text.set_text("")
		draw_current_item()

	def update_slider(exponent):
		if not updating_controls[0]:
			update_current_stiffness(10 ** float(exponent))

	def submit_stiffness(value):
		try:
			new_stiffness = float(value)
			new_exponent = np.log10(new_stiffness)
		except (TypeError, ValueError):
			status_text.set_text("Enter a positive numeric stiffness value.")
			draw_current_item()
			return
		if not np.isfinite(new_stiffness) or new_stiffness <= 0:
			status_text.set_text("Enter a positive finite stiffness value.")
			draw_current_item()
			return
		if not stiffness_slider.valmin <= new_exponent <= stiffness_slider.valmax:
			status_text.set_text("Value is outside the stiffness slider range.")
			draw_current_item()
			return
		status_text.set_text("")
		stiffness_slider.set_val(new_exponent)

	def save_and_advance(_event):
		item = items[index[0]]
		try:
			_write_corrected_spectrum(
				item["output_path"],
				item["data"],
				item["baseline"],
				output_format,
				item["signal_type"],
				output_units,
			)
		except (OSError, ValueError) as error:
			status_text.set_text(f"Save failed: {error}")
			figure.canvas.draw_idle()
			return
		saved_paths.append(item["output_path"])
		index[0] += 1
		if index[0] == len(items):
			plt.close(figure)
			return
		draw_current_item()

	stiffness_slider.on_changed(update_slider)
	stiffness_input.on_submit(submit_stiffness)
	save_button.on_clicked(save_and_advance)
	setattr(figure, "_batch_save_button", save_button)
	setattr(figure, "_batch_save_and_advance", save_and_advance)
	draw_current_item()
	plt.show()
	for item in items[index[0]:]:
		if item["output_path"] not in saved_paths:
			failures.append((item["input_path"], "Not saved before batch review window closed"))
	return saved_paths, failures


def _write_corrected_spectrum(
	output_path, data, baseline, output_format, signal_type, output_units
):
	units = signal_type if output_units == "source" else output_units
	signal, displayed_baseline = _convert_for_display(data, baseline, signal_type, units)
	corrected_data = np.column_stack((data[:, 0], signal - displayed_baseline))
	if output_format == "asp":
		ASP_Writer(output_path).write(corrected_data)
	else:
		np.savetxt(
			output_path,
			corrected_data,
			delimiter=",",
			header=f"wavenumber,baseline_corrected_{units}",
			comments="",
		)


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


def _choose_input_files():
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
	output_format="asp",
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
):
	"""Fit baselines and save corrected spectra for multiple ASP or CSV files.

	Returns (output_paths, failures), where failures contains (input_path, error)
	pairs for files that could not be processed.
	"""
	input_paths = [Path(path) for path in input_paths]
	if not input_paths:
		raise ValueError("At least one input file is required")
	if output_format not in ("asp", "csv"):
		raise ValueError("output_format must be 'asp' or 'csv'")
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
			if plot:
				display_units = resolved_signal_type if display_as == "source" else display_as
				_convert_for_display(data, baseline, resolved_signal_type, display_units)
			save_units = resolved_signal_type if output_units == "source" else output_units
			_convert_for_display(data, baseline, resolved_signal_type, save_units)
			items.append(
				{
					"input_path": input_path,
					"data": data,
					"signal_type": resolved_signal_type,
					"baseline": baseline,
					"stiffness": stiffness,
					"output_path": current_output_path,
				}
			)
		except (OSError, ValueError, RuntimeError) as error:
			failures.append((input_path, str(error)))

	if plot and items:
		output_paths, review_failures = show_batch_fits(
			items,
			stiffness=stiffness,
			baseline_points=baseline_points,
			fit_ranges=fit_ranges,
			ignore_ranges=ignore_ranges,
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
			_write_corrected_spectrum(
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
		help="Output path for a single input file",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		help="Directory for batch corrected spectra (defaults to each input's directory)",
	)
	parser.add_argument(
		"--output-format",
		choices=("csv", "asp"),
		default="asp",
		help="Corrected spectrum format (default: asp)",
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
	if not args.input:
		try:
			args.input = _choose_input_files()
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
