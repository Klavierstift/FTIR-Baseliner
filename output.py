"""Unit conversion and writing of baseline-corrected spectra."""


import numpy as np

from asp_writer import ASP_Writer


def write_corrected_spectrum(
	output_path, data, baseline, output_format, signal_type, output_units
):
	units = signal_type if output_units == "source" else output_units
	signal, displayed_baseline = convert_for_display(data, baseline, signal_type, units)
	corrected_data = np.column_stack((data[:, 0], signal - displayed_baseline))
	if output_format == "asp":
		ASP_Writer(output_path).write(corrected_data)
	else:
		# Same bytes as np.savetxt's default '%.18e' format, written in one call.
		body = "".join(f"{x:.18e},{y:.18e}\n" for x, y in corrected_data.tolist())
		with open(output_path, "w") as output:
			output.write(f"wavenumber,baseline_corrected_{units}\n")
			output.write(body)



def convert_for_display(data, baseline, signal_type, display_as):
	signal = np.asarray(data[:, 1], dtype=float)
	baseline = np.asarray(baseline, dtype=float)
	if signal_type == display_as:
		return signal, baseline
	if signal_type == "transmittance":
		if np.any(signal <= 0) or np.any(baseline <= 0):
			raise ValueError("Transmittance values must be positive to display absorbance")
		# Absorbance from percent transmittance (Beer-Lambert).
		with np.errstate(divide="ignore", invalid="ignore"):
			displayed_signal = -np.log10(signal / 100)
			displayed_baseline = -np.log10(baseline / 100)
	else:
		# Percent transmittance from absorbance: T = 100 * 10^-A.
		with np.errstate(over="ignore", invalid="ignore"):
			displayed_signal = 100 * np.power(10.0, -signal)
			displayed_baseline = 100 * np.power(10.0, -baseline)
	if not np.isfinite(displayed_signal).all() or not np.isfinite(displayed_baseline).all():
		raise ValueError("Spectrum values cannot be represented in the requested display units")
	return displayed_signal, displayed_baseline
