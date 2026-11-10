import csv
from io import StringIO
from pathlib import Path

import numpy as np


def detect_signal_type(data: np.ndarray) -> str:
	"""Infer transmittance from percent-scale values; otherwise use absorbance."""
	values = np.asarray(data, dtype=float)[:, 1]
	return "transmittance" if np.count_nonzero(values > 2) > len(values) / 2 else "absorbance"


class ASP_Reader:
	def __init__(self, file_path: str | Path):
		self.file_path = Path(file_path)
		self.signal_type = None

	def read(self) -> np.ndarray:
		text = self.file_path.read_text(encoding="utf-8-sig")
		lines = [line.strip() for line in text.splitlines() if line.strip()]
		data_lines = [line for line in lines if not line.startswith("#")]

		# CSV first; a file with no numeric rows falls back to the ASP layout below.
		spectrum_rows = []
		header_signal_type = None
		for row in csv.reader(StringIO(text)):
			if not row or not row[0].strip() or row[0].lstrip().startswith("#"):
				continue
			if not spectrum_rows:
				header_signal_type = _signal_type_from_header(row) or header_signal_type
			if len(row) != 2:
				if spectrum_rows:
					raise ValueError("CSV spectrum rows must contain two numeric columns")
				continue
			try:
				spectrum_rows.append([float(value.strip()) for value in row])
			except ValueError:
				if spectrum_rows:
					raise ValueError("CSV spectrum contains a nonnumeric row after spectral data")

		if spectrum_rows:
			data = np.asarray(spectrum_rows, dtype=float)
			if not np.isfinite(data).all():
				raise ValueError("CSV spectrum values must be finite")
			self.signal_type = header_signal_type or detect_signal_type(data)
			return data

		if len(data_lines) < 4:
			raise ValueError("Agilent ASP file is missing its header or spectral data")

		# Header is count, start, end, then two codes; intensities are the trailing lines.
		try:
			point_count_value, start_wavenumber, end_wavenumber = map(float, data_lines[:3])
			point_count = int(point_count_value)
		except ValueError as error:
			raise ValueError("Agilent ASP file contains invalid numeric values") from error

		if point_count_value != point_count or point_count < 1:
			raise ValueError("Agilent ASP point count must be a positive integer")
		if len(data_lines) < 3 + point_count:
			raise ValueError(
				f"Agilent ASP header declares {point_count} points, but the file contains only "
				f"{max(0, len(data_lines) - 3)} values after the header"
			)
		# Read from the end so any lines before the data block are skipped.
		try:
			intensities = np.asarray(
				[float(value) for value in data_lines[-point_count:]], dtype=float
			)
		except ValueError as error:
			raise ValueError("Agilent ASP spectral data contains invalid numeric values") from error
		if not np.isfinite([start_wavenumber, end_wavenumber]).all():
			raise ValueError("Agilent ASP wavenumber bounds must be finite")
		if not np.isfinite(intensities).all():
			raise ValueError("Agilent ASP intensities must be finite")

		wavenumbers = np.linspace(start_wavenumber, end_wavenumber, point_count)
		data = np.column_stack((wavenumbers, intensities))
		self.signal_type = detect_signal_type(data)
		return data


def _signal_type_from_header(row: list[str]) -> str | None:
	labels = {"".join(cell.lower().split()) for cell in row}
	if labels & {"%t", "%transmittance", "transmittance", "transmission"}:
		return "transmittance"
	if labels & {"a", "abs", "absorbance"}:
		return "absorbance"
	return None
