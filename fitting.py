"""Baseline fitting for two-column FTIR spectra."""


import numpy as np
from scipy.interpolate import CubicHermiteSpline

from asp_reader import detect_signal_type
from tension_fit import fit_baseline_cloth


def fit_asp_spectrum(
	data,
	signal_type="auto",
	baseline_points=226,
	stiffness=1e-5,
	fit_ranges=None,
	ignore_ranges=None,
	solver="newton",
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
		# Negate so absorption bands point upward; undone on return.
		fit_data[:, 1] *= -1
		invert_baseline = True
	else:
		raise ValueError("signal_type must be 'absorbance' or 'transmittance'")

	fit_ranges = validate_ranges(fit_ranges, "fit_ranges")
	ignore_ranges = validate_ranges(ignore_ranges, "ignore_ranges")
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
		# Knot ceilings: the lowest point in the segments touching each knot.
		caps = np.empty(len(indices), dtype=float)
		caps[0] = segment_minima[0]
		caps[-1] = segment_minima[-1]
		caps[1:-1] = np.minimum(segment_minima[:-1], segment_minima[1:])
		coarse_data = np.column_stack((selected_data[indices, 0], caps))

	# Normalizing by the selected range makes stiffness comparable across spectra.
	coarse_baseline = fit_baseline_cloth(
		coarse_data,
		stiffness=stiffness,
		signal_amplitude=np.ptp(selected_data[:, 1]),
		solver=solver,
	)
	# np.interp needs ascending x; the knots may not be sorted.
	order = np.argsort(coarse_data[:, 0])
	baseline = np.interp(
		data[:, 0], coarse_data[order, 0], coarse_baseline[order]
	)
	# Replace the fit inside ignored regions with a smooth bridge.
	if ignore_ranges:
		baseline = _smooth_ignored_bridges(fit_data, baseline, ignore_ranges)

	return -baseline if invert_baseline else baseline



def validate_ranges(ranges, name):
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
	# Merge overlapping ignore ranges so each gap is bridged once.
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
			# Clamp slopes so the bridge cannot overshoot the spectrum.
			slope_norm = np.hypot(left_slope / secant, right_slope / secant)
			if slope_norm > 3:
				limiter = 3 / slope_norm
				left_slope *= limiter
				right_slope *= limiter

		# Cubic Hermite bridge matching the fitted values and slopes at the gap edges.
		bridge = CubicHermiteSpline(
			[left_x, right_x],
			[left_y, right_y],
			[left_slope, right_slope],
		)
		fitted[left_index : right_index + 1] = bridge(
			wavenumbers[left_index : right_index + 1]
		)

	# Undo the sort so results align with the caller's point order.
	unsorted = np.empty_like(fitted)
	unsorted[order] = fitted
	return unsorted
