# FTIR Baseliner

FTIR Baseliner reads a spectrum, estimates a smooth baseline with a constrained cloth-fitting algorithm, and saves the baseline at the original spectrum's wavenumber points. During fitting, the baseline is constrained below the transformed signal; for absorbance this is below the measured spectrum, while for transmittance the reported baseline is above the measured `%T` spectrum. The fit balances raising its area against a curvature penalty, and `--stiffness` controls how strongly it resists bending.

## Requirements

Use Python 3.10 or newer. Install the dependencies with:

```powershell
python -m pip install numpy scipy matplotlib
```

## Run

Start the program and choose an input file from the picker:

```powershell
python main.py
```

Or provide an ASP or CSV file directly:

```powershell
python main.py samples/ABS-1.csv
```

The plot opens by default with the measured spectrum, baseline, and baseline-corrected spectrum. For cloth fits, adjust stiffness with the logarithmic slider or enter a positive value in the numeric field. The selected fit is saved after the plot closes.

By default, the baseline is saved beside the input as `<input-name>_baseline.asp`. Use `--output` to choose another path or `--output-format csv` to save a two-column CSV instead.

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
