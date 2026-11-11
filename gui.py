"""Matplotlib review windows for single and batch baseline fits."""


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import Button, Slider, SpanSelector, TextBox

from fitting import fit_asp_spectrum, validate_ranges
from output import convert_for_display, write_corrected_spectrum


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
	display_data, display_baseline = convert_for_display(data, baseline, signal_type, display_as)
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
	for index, (lower, upper) in enumerate(validate_ranges(fit_ranges, "fit_ranges")):
		for target_axis in (axis, fitted_axis) if show_fitted_spectrum else (axis,):
			target_axis.axvspan(
				lower,
				upper,
				color="tab:green",
				alpha=0.1,
				label="Fitting region" if index == 0 else None,
			)
	for index, (lower, upper) in enumerate(validate_ranges(ignore_ranges, "ignore_ranges")):
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
	# Reversed x-axis, the usual orientation for IR spectra.
	axis.set_xlim(float(np.max(data[:, 0])), float(np.min(data[:, 0])))
	axis.grid(True, alpha=0.25)
	axis.legend()
	if show_fitted_spectrum:
		fitted_axis.set_xlim(float(np.max(data[:, 0])), float(np.min(data[:, 0])))
		fitted_axis.grid(True, alpha=0.25)
		fitted_axis.legend()
	# Single-element lists let the slider callbacks update results in place.
	current_baseline = [np.asarray(baseline, dtype=float)]
	current_stiffness = [stiffness]
	if baseline_fitter is not None and stiffness is not None:
		# Bottom strip is reserved for the slider and value box.
		figure.tight_layout(rect=(0, 0.15, 1, 1))
		# Slider spans at least 1e-7 to 1e-3, widened around the starting stiffness.
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
				_, new_display_baseline = convert_for_display(
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
		# Keep widget references alive; matplotlib otherwise garbage-collects them.
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
	solver,
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

	# Same slider range rule as the single-spectrum window.
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
	# Clears the ignored ranges for the current spectrum only.
	clear_axis = figure.add_axes((0.81, 0.015, 0.16, 0.045))
	# Shared state for the button and slider callbacks.
	index = [0]
	saved_paths = []
	failures = []
	# Artists this window draws on the top axis; removed on each redraw.
	drawn_artists = []
	status_text = figure.text(0.13, 0.025, "", fontsize=9, color="tab:red")
	save_button = Button(button_axis, "Save / Next")
	clear_button = Button(clear_axis, "Clear ignored")
	# Set while draw_current_item moves the slider, so its callback does not refit.
	updating_controls = [False]

	def draw_current_item():
		item = items[index[0]]
		data = item["data"]
		display_units = item["signal_type"] if display_as == "source" else display_as
		display_data, display_baseline = convert_for_display(
			data, item["baseline"], item["signal_type"], display_units
		)
		# Remove only this window's artists; the span selector must stay attached.
		for artist in drawn_artists:
			artist.remove()
		drawn_artists.clear()
		# Explicit colors: recreated lines would otherwise take the next cycle color on each redraw.
		drawn_artists.extend(axis.plot(data[:, 0], display_data, color="tab:blue", label="Spectrum"))
		drawn_artists.extend(axis.plot(data[:, 0], display_baseline, color="tab:orange", label="Baseline fit"))
		axis.set_ylabel("Transmittance (%)" if display_units == "transmittance" else "Absorbance")
		axis.set_xlabel("Wavenumber (cm-1)")
		axis.set_xlim(float(np.max(data[:, 0])), float(np.min(data[:, 0])))
		axis.grid(True, alpha=0.25)
		for range_index, (lower, upper) in enumerate(validate_ranges(fit_ranges, "fit_ranges")):
			drawn_artists.append(
				axis.axvspan(
					lower,
					upper,
					color="tab:green",
					alpha=0.1,
					label="Fitting region" if range_index == 0 else None,
				)
			)
		for range_index, (lower, upper) in enumerate(item["ignore_ranges"]):
			drawn_artists.append(
				axis.axvspan(
					lower,
					upper,
					color="tab:red",
					alpha=0.12,
					label="Ignored region" if range_index == 0 else None,
				)
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
			for range_index, (lower, upper) in enumerate(validate_ranges(fit_ranges, "fit_ranges")):
				corrected_axis.axvspan(
					lower,
					upper,
					color="tab:green",
					alpha=0.1,
					label="Fitting region" if range_index == 0 else None,
				)
			for range_index, (lower, upper) in enumerate(item["ignore_ranges"]):
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

	def refit_current_item(new_stiffness, new_ignore_ranges):
		# Shared by stiffness changes and range selection; the item is unchanged on failure.
		item = items[index[0]]
		try:
			new_baseline = fit_asp_spectrum(
				item["data"],
				signal_type=item["signal_type"],
				baseline_points=baseline_points,
				stiffness=new_stiffness,
				fit_ranges=fit_ranges,
				ignore_ranges=new_ignore_ranges,
				solver=solver,
			)
			display_units = item["signal_type"] if display_as == "source" else display_as
			# Checks that the result can be displayed; failures show as a fit error.
			convert_for_display(item["data"], new_baseline, item["signal_type"], display_units)
		except (ValueError, RuntimeError) as error:
			status_text.set_text(f"Fit failed: {error}")
			figure.canvas.draw_idle()
			return
		item["baseline"] = new_baseline
		item["stiffness"] = new_stiffness
		item["ignore_ranges"] = list(new_ignore_ranges)
		status_text.set_text("")
		draw_current_item()

	def update_current_stiffness(new_stiffness):
		refit_current_item(new_stiffness, items[index[0]]["ignore_ranges"])

	def add_ignored_range(lower, upper):
		# Ignores accidental clicks; a real selection spans at least 1 cm-1.
		if upper - lower < 1.0:
			return
		item = items[index[0]]
		refit_current_item(
			item["stiffness"],
			item["ignore_ranges"] + [(float(lower), float(upper))],
		)

	def clear_ignored(_event):
		item = items[index[0]]
		if item["ignore_ranges"]:
			refit_current_item(item["stiffness"], [])

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
			write_corrected_spectrum(
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
	clear_button.on_clicked(clear_ignored)
	# Dragging across the top plot adds an ignored range and refits the current spectrum.
	range_selector = SpanSelector(
		axis,
		add_ignored_range,
		"horizontal",
		button=MouseButton.LEFT,
		useblit=True,
		props={"facecolor": "tab:red", "alpha": 0.2},
		interactive=False,
	)

	def remove_ignored_range_at(event):
		# Right-click inside an ignored range removes only that range.
		if event.button != MouseButton.RIGHT or event.inaxes is not axis or event.xdata is None:
			return
		item = items[index[0]]
		for range_index, (lower, upper) in enumerate(item["ignore_ranges"]):
			if lower <= event.xdata <= upper:
				kept = item["ignore_ranges"][:range_index] + item["ignore_ranges"][range_index + 1 :]
				refit_current_item(item["stiffness"], kept)
				return

	figure.canvas.mpl_connect("button_press_event", remove_ignored_range_at)
	axis.set_title("Drag to ignore a range; right-click a range to remove it", fontsize=9, color="gray")
	setattr(figure, "_range_selector", range_selector)
	setattr(figure, "_batch_save_button", save_button)
	setattr(figure, "_batch_clear_button", clear_button)
	setattr(figure, "_batch_save_and_advance", save_and_advance)
	draw_current_item()
	plt.show()
	# Items never saved before the window closed count as failures.
	for item in items[index[0]:]:
		if item["output_path"] not in saved_paths:
			failures.append((item["input_path"], "Not saved before batch review window closed"))
	return saved_paths, failures
