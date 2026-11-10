"""Fit smooth FTIR baselines using a constrained cloth optimizer."""

from functools import lru_cache

import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, diags


def _band_storage(hessian, bandwidth=2):
	"""Pack a banded sparse matrix into the layout expected by solve_banded."""
	n = hessian.shape[0]
	coo = hessian.tocoo()
	band = np.zeros((2 * bandwidth + 1, n))
	np.add.at(band, (bandwidth + coo.row - coo.col, coo.col), coo.data)
	return band


def _solve_free_block(band, free, rhs, ridge):
	"""Solve the free-variable block of a banded system; fixed variables give zero steps."""
	bandwidth = (band.shape[0] - 1) // 2
	n = band.shape[1]
	masked = np.zeros_like(band)
	for row in range(band.shape[0]):
		offset = row - bandwidth
		partner = np.zeros(n, dtype=bool)
		if offset >= 0:
			partner[: n - offset] = free[offset:]
		else:
			partner[-offset:] = free[: n + offset]
		masked[row] = np.where(free & partner, band[row], 0.0)
	# Fixed variables get identity rows so their steps come out as zero.
	masked[bandwidth] = np.where(free, band[bandwidth] + ridge, 1.0)
	return solve_banded((bandwidth, bandwidth), masked, np.where(free, rhs, 0.0))


def _minimize_bounded_quadratic(hessian, linear, upper, max_iter=1000, tol=1e-10):
	"""Minimize 0.5 x'Hx - linear'x over 0 <= x <= upper with projected Newton steps."""
	# Zero is feasible for 0 <= x <= upper, so it is a valid start.
	x = np.zeros_like(upper)
	# Gradient roundoff grows with the largest Hessian entry, so the floor scales with it.
	diagonal = hessian.diagonal()
	tol = max(tol, 1e-13 * float(diagonal.max()))
	band = _band_storage(hessian)
	for _ in range(max_iter):
		gradient = hessian @ x - linear
		if np.max(np.abs(np.clip(x - gradient, 0, upper) - x)) <= tol:
			return x
		# Variables at a bound whose gradient pushes outward stay fixed this step.
		lower_active = (x <= 0) & (gradient >= 0)
		upper_active = (x >= upper) & (gradient <= 0)
		free_mask = ~(lower_active | upper_active)
		step = np.zeros_like(x)
		if free_mask.any():
			# Tiny ridge keeps the reduced system solvable when it is nearly singular.
			ridge = 1e-12 * max(float(diagonal[free_mask].max()), 1e-300)
			step = _solve_free_block(band, free_mask, -gradient, ridge)
		current = 0.5 * x @ (hessian @ x) - linear @ x
		scale = 1.0
		while True:
			candidate = np.clip(x + scale * step, 0, upper)
			candidate_value = 0.5 * candidate @ (hessian @ candidate) - linear @ candidate
			# Armijo test; the scale floor ends the search when roundoff hides the decrease.
			if candidate_value <= current + 1e-4 * (gradient @ (candidate - x)) or scale < 1e-12:
				break
			scale *= 0.5
		if scale < 1e-12:
			break
		x = candidate
	raise RuntimeError("Cloth baseline fit did not converge")


def _minimize_lbfgsb(hessian, linear, upper):
	"""Minimize the same bounded quadratic with SciPy's L-BFGS-B."""
	def objective(x):
		hx = hessian @ x
		return 0.5 * x @ hx - linear @ x, hx - linear

	result = minimize(
		objective,
		x0=np.zeros_like(upper),
		method="L-BFGS-B",
		jac=True,
		bounds=list(zip(np.zeros_like(upper), upper)),
		options={"maxiter": 1000, "ftol": 1e-11, "gtol": 1e-8},
	)
	if not result.success and result.status != 1:
		raise RuntimeError(f"Cloth baseline fit failed: {result.message}")
	return result.x


def _split_spectrum_input(wavenumbers, spectrum):
	if spectrum is None:
		data = np.asarray(wavenumbers, dtype=float)
		if data.ndim != 2 or data.shape[1] != 2:
			raise ValueError("Provide an (n, 2) array or list of [wavenumber, intensity] pairs.")
		return data[:, 0], data[:, 1]
	return np.asarray(wavenumbers, dtype=float), np.asarray(spectrum, dtype=float)


@lru_cache(maxsize=16)
def _stiffness_free_operator(wavenumber_key):
	"""Return trapezoid weights and the unscaled curvature Hessian for an ascending grid."""
	x = np.asarray(wavenumber_key, dtype=float)
	# Positions in [0, 1] keep the stiffness scale independent of the wavenumber span.
	position = (x - x[0]) / (x[-1] - x[0])
	intervals = np.diff(position)
	left_widths = intervals[:-1]
	right_widths = intervals[1:]
	# Three-point second-difference stencil for non-uniform spacing.
	curvature_left = 2 / (left_widths * (left_widths + right_widths))
	curvature_middle = -2 / (left_widths * right_widths)
	curvature_right = 2 / (right_widths * (left_widths + right_widths))
	curvature_weights = (left_widths + right_widths) / 2
	# Trapezoid weights: the linear term that rewards raising the baseline.
	area_weights = np.empty_like(position)
	area_weights[0] = intervals[0] / 2
	area_weights[-1] = intervals[-1] / 2
	area_weights[1:-1] = (intervals[:-1] + intervals[1:]) / 2

	# Sparse second-difference matrix D; the Hessian is stiffness * D'WD.
	count = len(position) - 2
	rows = np.repeat(np.arange(count), 3)
	columns = np.column_stack((np.arange(count), np.arange(count) + 1, np.arange(count) + 2)).ravel()
	coefficients = np.column_stack((curvature_left, curvature_middle, curvature_right)).ravel()
	second_difference = csr_matrix((coefficients, (rows, columns)), shape=(count, len(position)))
	base_hessian = (second_difference.T @ diags(curvature_weights) @ second_difference).tocsr()
	area_weights.setflags(write=False)
	return area_weights, base_hessian


def fit_baseline_cloth(
	wavenumbers, spectrum=None, stiffness=1e-5, signal_amplitude=None, solver="newton"
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
	if solver not in ("newton", "lbfgsb"):
		raise ValueError("solver must be 'newton' or 'lbfgsb'")

	differences = np.diff(x)
	descending = bool(np.all(differences < 0))
	if not descending and not np.all(differences > 0):
		raise ValueError("Wavenumbers must be strictly monotonic without duplicates.")
	# Solve in ascending order; the baseline is reversed back before returning.
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

	# Values scaled so the spectrum minimum is 0 and its range is 1.
	values = (y - data_minimum) / signal_amplitude
	area_weights, base_hessian = _stiffness_free_operator(tuple(x.tolist()))
	hessian = stiffness * base_hessian
	if solver == "newton":
		cloth = _minimize_bounded_quadratic(hessian, area_weights, values)
	else:
		cloth = _minimize_lbfgsb(hessian, area_weights, values)
	baseline = data_minimum + signal_amplitude * cloth
	return baseline[::-1].copy() if descending else baseline

