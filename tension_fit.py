"""Fit a lower, piecewise-linear FTIR baseline with limited segment turns.

Each adjacent pair of wavenumbers defines one segment. Its heading is limited
relative to the previous segment's heading, rather than independently against
the x-axis. Only NumPy is required.
"""

import numpy as np
from math import atan2
from scipy.optimize import minimize


def _split_spectrum_input(wavenumbers, spectrum):
	if spectrum is None:
		data = np.asarray(wavenumbers, dtype=float)
		if data.ndim != 2 or data.shape[1] != 2:
			raise ValueError("Provide an (n, 2) array or list of [wavenumber, intensity] pairs.")
		return data[:, 0], data[:, 1]
	return np.asarray(wavenumbers, dtype=float), np.asarray(spectrum, dtype=float)


def fit_baseline(
	wavenumbers, spectrum=None, rotation_freedom=10.0, signal_amplitude=None
):
	"""Return the highest baseline permitted by the rotation constraint.

	Parameters
	----------
	wavenumbers : array-like
		Either an (n, 2) array/list of wavenumber and intensity pairs, or the
		wavenumber coordinates when ``spectrum`` is provided separately.
	spectrum : array-like, shape (n,), optional
		Intensity values corresponding to separate ``wavenumbers``.
	rotation_freedom : float, default 10
		Percentage from 0 to 100. Each segment may turn by at most
		0.9 * rotation_freedom degrees from the preceding segment. The first
		segment uses the x-axis as its reference heading. Both axes are scaled
		to their full ranges before angles are calculated.
	signal_amplitude : float, optional
		Intensity range used to normalize the y-axis. Must be at least the range
		of the provided spectrum. Use the original signal's range when fitting a
		conservative coarse-grid envelope.

	Notes
	-----
	The baseline is raised from a constant feasible line while respecting the
	maximum heading change at every shared segment endpoint. Linear
	interpolation of both baseline and spectrum cannot introduce crossings
	between samples.
	Peaks are assumed to point upward (e.g. absorbance, not transmittance).
	The array/list input can be passed directly from ``ASP_Reader.read()``.
	"""
	x, y = _split_spectrum_input(wavenumbers, spectrum)
	freedom = float(rotation_freedom)
	if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.size < 2:
		raise ValueError("Provide matching one-dimensional arrays of length >= 2.")
	if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
		raise ValueError("Wavenumbers and spectrum must contain only finite values.")
	if not np.isfinite(freedom) or not 0 <= freedom <= 100:
		raise ValueError("rotation_freedom must be between 0 and 100 percent.")

	differences = np.diff(x)
	descending = bool(np.all(differences < 0))
	if not descending and not np.all(differences > 0):
		raise ValueError("Wavenumbers must be strictly monotonic without duplicates.")
	if descending:
		x, y = x[::-1], y[::-1]

	amplitude = np.ptp(y)
	if signal_amplitude is not None:
		signal_amplitude = float(signal_amplitude)
		if not np.isfinite(signal_amplitude) or signal_amplitude < amplitude:
			raise ValueError("signal_amplitude must be finite and at least the spectrum range")
		amplitude = signal_amplitude
	if freedom == 100:
		baseline = y.copy()
	elif amplitude == 0:
		baseline = y.copy()
	elif freedom == 0:
		baseline = np.full_like(y, np.min(y))
	else:
		position = (x - x[0]) / (x[-1] - x[0])
		ceiling = (y - np.min(y)) / amplitude
		step_widths = np.diff(position)
		maximum_turn = np.deg2rad(0.9 * freedom)
		baseline_scaled = np.zeros_like(ceiling)

		def turn_at(vertex):
			previous_angle = atan2(
				baseline_scaled[vertex] - baseline_scaled[vertex - 1],
				step_widths[vertex - 1],
			)
			next_angle = atan2(
				baseline_scaled[vertex + 1] - baseline_scaled[vertex],
				step_widths[vertex],
			)
			return next_angle - previous_angle

		def can_raise(index, value):
			previous_value = baseline_scaled[index]
			baseline_scaled[index] = value
			allowed = True
			if index <= 1:
				allowed = abs(
					atan2(
					baseline_scaled[1] - baseline_scaled[0], step_widths[0]
					)
				) <= maximum_turn + 1e-12
			first_vertex = max(1, index - 1)
			last_vertex = min(len(baseline_scaled) - 2, index + 1)
			for vertex in range(first_vertex, last_vertex + 1):
				if abs(turn_at(vertex)) > maximum_turn + 1e-12:
					allowed = False
					break
			baseline_scaled[index] = previous_value
			return allowed

		converged = False
		for _ in range(max(120, min(240, len(y)))):
			largest_increase = 0.0
			for indices in (range(len(y)), range(len(y) - 1, -1, -1)):
				for index in indices:
					previous_value = baseline_scaled[index]
					lower = previous_value
					upper = ceiling[index]
					if upper <= lower + 1e-10:
						continue
					if can_raise(index, upper):
						baseline_scaled[index] = upper
					else:
						for _ in range(28):
							middle = (lower + upper) / 2
							if can_raise(index, middle):
								lower = middle
							else:
								upper = middle
						baseline_scaled[index] = lower
					largest_increase = max(
							largest_increase, baseline_scaled[index] - previous_value
					)
			if largest_increase <= 1e-4:
				converged = True
				break
		if not converged:
			raise RuntimeError("Baseline fit did not converge within the iteration limit.")
		baseline = np.min(y) + amplitude * baseline_scaled

	return baseline[::-1].copy() if descending else baseline


def fit_baseline_cloth(
	wavenumbers, spectrum=None, stiffness=1e-5, signal_amplitude=None
):
	"""Fit a smooth baseline by pushing a flexible cloth up against the data.

	The cloth is pulled upward while curvature stiffness resists bending. The
	spectrum acts as a collision ceiling, so the fitted baseline cannot cross
	it. A larger stiffness produces a smoother, less peak-following baseline.
	"""
	x, y = _split_spectrum_input(wavenumbers, spectrum)
	stiffness = float(stiffness)
	if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.size < 3:
		raise ValueError("Provide matching one-dimensional arrays of length >= 3.")
	if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
		raise ValueError("Wavenumbers and spectrum must contain only finite values.")
	if not np.isfinite(stiffness) or stiffness <= 0:
		raise ValueError("stiffness must be finite and greater than zero")

	differences = np.diff(x)
	descending = bool(np.all(differences < 0))
	if not descending and not np.all(differences > 0):
		raise ValueError("Wavenumbers must be strictly monotonic without duplicates.")
	if descending:
		x, y = x[::-1], y[::-1]

	data_minimum = float(np.min(y))
	amplitude = float(np.ptp(y))
	if amplitude == 0:
		return y[::-1].copy() if descending else y.copy()
	if signal_amplitude is not None:
		signal_amplitude = float(signal_amplitude)
		if not np.isfinite(signal_amplitude) or signal_amplitude < amplitude:
			raise ValueError("signal_amplitude must be finite and at least the spectrum range")
	else:
		signal_amplitude = amplitude

	position = (x - x[0]) / (x[-1] - x[0])
	values = (y - data_minimum) / signal_amplitude
	intervals = np.diff(position)
	left_widths = intervals[:-1]
	right_widths = intervals[1:]
	curvature_left = 2 / (left_widths * (left_widths + right_widths))
	curvature_middle = -2 / (left_widths * right_widths)
	curvature_right = 2 / (right_widths * (left_widths + right_widths))
	curvature_weights = (left_widths + right_widths) / 2
	area_weights = np.empty_like(position)
	area_weights[0] = intervals[0] / 2
	area_weights[-1] = intervals[-1] / 2
	area_weights[1:-1] = (intervals[:-1] + intervals[1:]) / 2

	def objective(cloth):
		curvature = (
			curvature_left * cloth[:-2]
			+ curvature_middle * cloth[1:-1]
			+ curvature_right * cloth[2:]
		)
		weighted_curvature = curvature_weights * curvature
		gradient = -area_weights.copy()
		gradient[:-2] += stiffness * curvature_left * weighted_curvature
		gradient[1:-1] += stiffness * curvature_middle * weighted_curvature
		gradient[2:] += stiffness * curvature_right * weighted_curvature
		energy = 0.5 * stiffness * np.dot(curvature_weights, curvature**2)
		energy -= np.dot(area_weights, cloth)
		return energy, gradient

	result = minimize(
		objective,
		x0=np.zeros_like(values),
		method="L-BFGS-B",
		jac=True,
		bounds=list(zip(np.zeros_like(values), values)),
		options={"maxiter": 1000, "ftol": 1e-11, "gtol": 1e-8},
	)
	if not result.success and result.status != 1:
		raise RuntimeError(f"Cloth baseline fit failed: {result.message}")
	baseline = data_minimum + signal_amplitude * result.x
	return baseline[::-1].copy() if descending else baseline

