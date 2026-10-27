# FTIR Baseliner

FTIR Baseliner reads a spectrum, estimates a smooth baseline with a constrained cloth-fitting algorithm, and saves the baseline at the original spectrum's wavenumber points. During fitting, the baseline is constrained below the transformed signal; for absorbance this is below the measured spectrum, while for transmittance the reported baseline is above the measured `%T` spectrum. The fit balances raising its area against a curvature penalty, and `--stiffness` controls how strongly it resists bending.

## Requirements

Use Python 3.10 or newer. Install the dependencies with:

```powershell
python -m pip install numpy scipy matplotlib
```

## Run

Start the program and choose one or more input files from the picker:

```powershell
python main.py
```

Or provide one or more ASP or CSV files directly:

```powershell
python main.py samples/ABS-1.csv
python main.py samples/ABS-1.csv samples/ABS-2.csv --no-plot --output-dir results
```

The plot opens by default with the measured spectrum, baseline, and baseline-corrected spectrum. For cloth fits, adjust stiffness with the logarithmic slider or enter a positive value in the numeric field. In a multi-file run, one window stays open while files are reviewed. Click **Save / Next** to save the current baseline and show the next file; the final button is **Save / Done**. Closing the window early leaves unconfirmed files unsaved and reports them. Use `--no-plot` for unattended batch processing.

By default, each baseline is saved beside its input as `<input-name>_baseline.asp`. Use `--output` to choose a different path for a single input, or `--output-dir` to put batch outputs in one directory. Use `--output-format csv` to save two-column CSV files instead.

## Input Formats

- Native Agilent-style ASP files with a linear wavenumber axis.
- CSV files with two numeric columns: wavenumber and intensity. Vendor metadata and column headers before the data are supported.

Signal type is detected from CSV unit headers such as `%T` or `A`. For unlabeled files, the program uses a value-scale heuristic. Specify `--signal-type absorbance` or `--signal-type transmittance` if detection needs to be overridden.

## Options

| Option | Description |
| --- | --- |
| `--stiffness VALUE` | Cloth curvature resistance; default is `9e-5`. Lower values allow more bending; higher values produce a smoother baseline. |
| `--baseline-points COUNT` | Number of coarse baseline points; default is `226`. |
| `--signal-type TYPE` | `auto`, `absorbance`, or `transmittance`; default is `auto`. |
| `--display-as UNITS` | Plot in `source`, `absorbance`, or `transmittance` units; default is `source`. Conversion affects the plot only, not the saved baseline. |
| `--fit-range MIN MAX` | Restrict baseline fitting to this wavenumber range. Repeat to specify multiple ranges. |
| `--ignore-range MIN MAX` | Exclude this range from fitting and smoothly bridge the baseline across it. Repeat for multiple ranges. |
| `--no-plot` | Disable plotting. Plotting is on by default. |
| `--no-show-baseline-corrected` | Hide the baseline-corrected panel while keeping the main plot. It is shown by default. |
| `--output PATH` | Set the baseline output path. |
| `--output-dir PATH` | Set the directory for outputs from a batch run. |
| `--output-format FORMAT` | Choose `asp` or `csv`; default is `asp`. |
| `-h`, `--help` | Show all command-line options. |

Examples:

```powershell
# Save CSV output without opening the plot
python main.py samples/ABS-1.csv --no-plot --output-format csv

# Plot as absorbance, even when the source file contains transmittance
python main.py samples/ABS-1.csv --display-as absorbance

# Fit only the specified regions and skip the range between them
python main.py samples/ABS-1.csv --fit-range 650 1800 --fit-range 2800 4000 --ignore-range 1800 2800

# Process several files without opening an individual plot for each
python main.py samples/ABS-1.csv samples/ABS-2.csv --no-plot --output-dir results
```

## Python API

```python
from asp_reader import ASP_Reader
from main import fit_asp_spectrum

spectrum = ASP_Reader("samples/ABS-1.csv").read()
baseline = fit_asp_spectrum(
    spectrum,
    signal_type="auto",
    stiffness=9e-5,
)
```

`baseline` is a one-dimensional NumPy array aligned with the input rows. The fitter is cloth-only; the previous rotation-based fitter is no longer included.

For multiple input files, use `fit_batch_spectra()`. It returns `(output_paths, failures)`, where each failure contains the input path and an error message:

```python
from main import fit_batch_spectra

outputs, failures = fit_batch_spectra(
    ["samples/ABS-1.csv", "samples/ABS-2.csv"],
    output_dir="results",
    plot=False,
)
```
