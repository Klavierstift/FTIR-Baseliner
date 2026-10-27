"""Fit smooth FTIR baselines using a constrained cloth optimizer."""

import numpy as np
from scipy.optimize import minimize


def _split_spectrum_input(wavenumbers, spectrum):
	if spectrum is None:
		data = np.asarray(wavenumbers, dtype=float)
		if data.ndim != 2 or data.shape[1] != 2:
			raise ValueError("Provide an (n, 2) array or list of [wavenumber, intensity] pairs.")
		return data[:, 0], data[:, 1]
	return np.asarray(wavenumbers, dtype=float), np.asarray(spectrum, dtype=float)


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

