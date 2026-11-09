# FTIR Baseliner

FTIR Baseliner reads a spectrum, estimates a smooth baseline with a constrained cloth-fitting algorithm, subtracts that baseline from the measured signal, and saves the baseline-corrected spectrum at the original wavenumber points. During fitting, the baseline is constrained below the transformed signal; for absorbance this is below the measured spectrum, while for transmittance the reported baseline is above the measured `%T` spectrum. The fit balances raising its area against a curvature penalty, and `--stiffness` controls how strongly it resists bending.

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

The plot opens by default with the measured spectrum, baseline, and baseline-corrected spectrum. For cloth fits, adjust stiffness with the logarithmic slider or enter a positive value in the numeric field. The selected fit is applied when the corrected spectrum is saved after the plot closes. In a multi-file run, one window stays open while files are reviewed. Click **Save / Next** to save the current corrected spectrum and show the next file; the final button is **Save / Done**. Closing the window early leaves unconfirmed files unsaved and reports them. Use `--no-plot` for unattended batch processing.

By default, each corrected spectrum is saved beside its input as `<input-name>_corrected.asp`. The saved intensity is `measured signal - baseline` in the input's source units; `--display-as` changes the plot only. Use `--output` to choose a different path for a single input, or `--output-dir` to put batch outputs in one directory. Use `--output-format csv` to save two-column CSV files instead.

## Input Formats

- Native Agilent-style ASP files with a linear wavenumber axis.
- CSV files with two numeric columns: wavenumber and intensity. Vendor metadata and column headers before the data are supported.

Signal type is detected from CSV unit headers such as `%T` or `A`. For unlabeled files, the program uses a value-scale heuristic. Specify `--signal-type absorbance` or `--signal-type transmittance` if detection needs to be overridden.

## How It Works

1. **Read the spectrum.** ASP files provide a point count and the endpoints of a linear wavenumber axis, followed by intensity values. CSV loading skips metadata and column labels until it reaches two-column numeric spectrum data. The input rows and their order are retained for the final output.
2. **Determine the signal orientation.** Absorbance peaks point upward, while transmittance absorption bands appear as downward dips. For fitting, transmittance intensities are sign-inverted so both signal types can use the same lower-envelope algorithm. The baseline is inverted back afterward, placing it below absorbance peaks or above the measured `%T` signal.
3. **Choose the fitting points.** `--fit-range` selects the wavenumber intervals that contribute to fitting; repeated ranges are combined. `--ignore-range` removes its intervals from fitting as well. If the spectrum is larger than `--baseline-points`, the program places coarse knots across the selected data and uses segment minima to form a conservative envelope for the cloth fit.
4. **Fit the cloth.** The optimizer raises a smooth baseline toward the spectrum without crossing its envelope. It maximizes baseline area while penalizing curvature, using bounded L-BFGS-B optimization. `--stiffness` sets the curvature penalty: smaller values allow more bending and can follow broader background changes, while larger values favor a smoother, flatter line.
5. **Restore full resolution.** The coarse baseline is linearly interpolated to every original wavenumber. For ignored intervals, the program replaces the interpolated section with a cubic Hermite bridge between the fitted values at its boundaries, limiting endpoint slopes to avoid sharp transitions.
6. **Review and save.** The plot can show the spectrum and baseline in source units or convert both for display as absorbance or transmittance. Display conversion does not change the saved corrected spectrum, which stays in the input's source units. In batch review, the same window advances through spectra; **Save / Next** writes the current corrected spectrum, and **Save / Done** writes the last one. Closing early leaves remaining spectra unsaved.

### Cloth Optimization Details

The cloth is a mathematical analogy, not a physical simulation. The optimizer first scales wavenumbers to the interval `[0, 1]` and shifts/scales intensity values using the signal minimum and amplitude. It then assigns one baseline value, `b`, to each coarse knot.

The baseline values are bounded by the data envelope:

$$
0 \leq b_j \leq y_j
$$

Here `y` is the scaled coarse envelope. These bounds prevent the fitted baseline from crossing the signal used for fitting. In transmittance mode, the signal has already been sign-inverted, so converting the result back places the baseline above the measured `%T` trace.

The optimizer minimizes an energy with an area reward and a curvature penalty:

$$
E(b) = \frac{\lambda}{2}\sum_i w_i\,\kappa_i(b)^2 - \sum_j a_j\,b_j
$$

`κᵢ(b)` is the local second derivative of the baseline, approximated from three neighboring knots. `wᵢ` and `aⱼ` are spacing-aware weights that account for the knot intervals when measuring curvature and area. The negative area term rewards raising the baseline; the squared-curvature term penalizes bending. `λ` is the stiffness parameter. L-BFGS-B minimizes this objective under the bounds above, using the analytic gradient computed by the fitter.

Consequently, a smaller stiffness allows the baseline to bend more and follow broad changes more closely. A larger stiffness favors a flatter baseline, but may leave a larger gap beneath broad spectral structure. The slider changes `λ` on a logarithmic scale because useful stiffness values span several orders of magnitude.

## Options

| Option | Description |
| --- | --- |
| `--stiffness VALUE` | Cloth curvature resistance; default is `9e-5`. Lower values allow more bending; higher values produce a smoother baseline. |
| `--baseline-points COUNT` | Number of coarse baseline points; default is `226`. |
| `--signal-type TYPE` | `auto`, `absorbance`, or `transmittance`; default is `auto`. |
| `--display-as UNITS` | Plot in `source`, `absorbance`, or `transmittance` units; default is `source`. Conversion affects the plot only; the corrected output remains in source units. |
| `--fit-range MIN MAX` | Restrict baseline fitting to this wavenumber range. Repeat to specify multiple ranges. |
| `--ignore-range MIN MAX` | Exclude this range from fitting and smoothly bridge the baseline across it. Repeat for multiple ranges. |
| `--no-plot` | Disable plotting. Plotting is on by default. |
| `--no-show-baseline-corrected` | Hide the baseline-corrected panel while keeping the main plot. It is shown by default. |
| `--output PATH` | Set the corrected-spectrum output path for a single input. |
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
