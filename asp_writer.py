from pathlib import Path

import numpy as np


class ASP_Writer:
	def __init__(self, file_path: str | Path):
		self.file_path = Path(file_path)

	def write(self, data: np.ndarray) -> None:
		data = np.asarray(data, dtype=float)
		if data.ndim != 2 or data.shape[1] != 2 or data.shape[0] < 2:
			raise ValueError("Spectrum data must contain at least two wavenumber/intensity pairs")
		if not np.isfinite(data).all():
			raise ValueError("Spectrum values must be finite")

		wavenumbers = data[:, 0]
		steps = np.diff(wavenumbers)
		if not (np.all(steps > 0) or np.all(steps < 0)):
			raise ValueError("ASP wavenumbers must be strictly monotonic")
		# The ASP header stores only start and end, so spacing must be uniform.
		if not np.allclose(steps, steps[0], rtol=1e-5, atol=1e-8):
			raise ValueError("ASP format requires evenly spaced wavenumbers")

		with self.file_path.open("w", encoding="ascii", newline="\n") as output:
			output.write(f"{len(data)}\n")
			output.write(f"{wavenumbers[0]:.10g}\n")
			output.write(f"{wavenumbers[-1]:.10g}\n")
			output.write("1\n2\n")
			for intensity in data[:, 1]:
				output.write(f"{intensity:.8f}\n")