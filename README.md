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

By default, each corrected spectrum is saved beside its input as `<input-name>_corrected.asp`. The saved intensity is `measured signal - baseline` in the input's source units; `--display-as` changes the plot only, and `--output-units` changes the saved units. Use `--output` to choose a different path for a single input, or `--output-dir` to put batch outputs in one directory. Use `--output-format csv` to save two-column CSV files instead.

## Input Formats

- Native Agilent-style ASP files with a linear wavenumber axis.
- CSV files with two numeric columns: wavenumber and intensity. Vendor metadata and column headers before the data are supported.

Signal type is detected from CSV unit headers such as `%T` or `A`. For unlabeled files, the program uses a value-scale heuristic. Specify `--signal-type absorbance` or `--signal-type transmittance` if detection needs to be overridden.

## How It Works

1. **Read the spectrum.** ASP files provide a point count and the endpoints of a linear wavenumber axis, followed by intensity values. CSV loading skips metadata and column labels until it reaches two-column numeric spectrum data. The input rows and their order are retained for the final output.
2. **Determine the signal orientation.** Absorbance peaks point upward, while transmittance absorption bands appear as downward dips. For fitting, transmittance intensities are sign-inverted so both signal types can use the same lower-envelope algorithm. The baseline is inverted back afterward, placing it below absorbance peaks or above the measured `%T` signal.
3. **Choose the fitting points.** `--fit-range` selects the wavenumber intervals that contribute to fitting; repeated ranges are combined. `--ignore-range` removes its intervals from fitting as well. A knot is a coarse point where the optimizer solves for one baseline value; the baseline between knots is interpolated. If the spectrum is larger than `--baseline-points`, the program places evenly spaced knots across the selected data. Each knot's ceiling is the lower of the minima of its two adjacent segments, so the cloth cannot cross the signal between knots. The [Cloth Optimization Details](#cloth-optimization-details) section explains the fit.
4. **Fit the cloth.** The optimizer raises a smooth baseline toward the spectrum without crossing its envelope. It maximizes baseline area while penalizing curvature, using a bounded projected Newton solver with an exact sparse Hessian. `--stiffness` sets the curvature penalty: smaller values allow more bending and can follow broader background changes, while larger values favor a smoother, flatter line.
5. **Restore full resolution.** The coarse baseline is linearly interpolated to every original wavenumber. For ignored intervals, the program replaces the interpolated section with a cubic Hermite bridge between the fitted values at its boundaries, limiting endpoint slopes to avoid sharp transitions.
6. **Review and save.** The plot can show the spectrum and baseline in source units or convert both for display as absorbance or transmittance. Display conversion does not change the saved corrected spectrum. The saved spectrum is in the input's source units unless `--output-units` is set; in that case both the spectrum and baseline are converted before subtracting, so the saved values match the corrected residual in the chosen units. In batch review, the same window advances through spectra; **Save / Next** writes the current corrected spectrum, and **Save / Done** writes the last one. Closing early leaves remaining spectra unsaved.

### Cloth Optimization Details

The cloth is a mathematical analogy, not a physical simulation. A flexible sheet is pulled upward until it touches the spectrum from below, while its stiffness resists bending. The fit is a bounded quadratic minimization with four stages: build the ceiling, normalize the problem, minimize the energy, and verify convergence.

#### 1. Ceiling Construction

Before the cloth fit, `main.py` builds the ceiling values from the selected points:

- If the selected spectrum has more points than `--baseline-points`, the points are split into segments between evenly spaced knots.
- Each interior knot takes the lower of the minima of its two adjacent segments. The end knots take the minimum of their single adjacent segment.
- If the spectrum has no more points than `--baseline-points`, every point is its own knot and its own ceiling.

Because each ceiling is at or below the signal across its span, the cloth stays under the measured data.

#### 2. Normalization

The optimizer maps the wavenumbers to $t_i \in [0, 1]$ and scales the knot ceilings to

$$
y_j = \frac{c_j - c_{\min}}{\Delta}
$$

where $c_j$ is the knot ceiling, $c_{\min}$ is the lowest ceiling, and $\Delta$ is the intensity range of the selected data. Normalizing removes the dependence on the wavenumber span and intensity units, so the same `--stiffness` produces comparable behavior across spectra.

Each knot has one baseline value, $b_j$, bounded by the scaled ceiling:

$$
0 \leq b_j \leq y_j
$$

These bounds prevent the fitted baseline from crossing the signal used for fitting. In transmittance mode, the signal has already been sign-inverted, so converting the result back places the baseline above the measured `%T` trace.

#### 3. Energy

Let $h_i = t_{i+1} - t_i$ be the knot spacings. The curvature at interior knot $i$ is estimated from knots $i$, $i+1$, and $i+2$, which keeps the estimate correct for uneven spacing:

$$
\kappa_i(b) = \frac{2}{h_i(h_i+h_{i+1})}\,b_i - \frac{2}{h_i h_{i+1}}\,b_{i+1} + \frac{2}{h_{i+1}(h_i+h_{i+1})}\,b_{i+2}
$$

The curvature weight is $w_i = (h_i + h_{i+1})/2$. The area weights $a_j$ are trapezoid weights, with each knot receiving half of each adjacent interval. The optimizer minimizes an energy with an area reward and a curvature penalty:

$$
E(b) = \frac{\lambda}{2}\sum_i w_i\,\kappa_i(b)^2 - \sum_j a_j\,b_j
$$

The negative area term rewards raising the baseline, and the squared-curvature term penalizes bending. $\lambda$ is the stiffness. On the normalized axis, this discrete energy approximates $\lambda\int_0^1 b''(t)^2\,dt - \int_0^1 b(t)\,dt$, so changing `--baseline-points` mainly changes resolution rather than the balance set by stiffness.

The energy is quadratic, so it can be written as $\tfrac{1}{2}b^\top H b - a^\top b$, with the sparse Hessian $H = \lambda D^\top W D$. Here $D$ holds the curvature stencil and $W$ is the diagonal matrix of curvature weights. The gradient is $Hb - a$.

#### 4. Projected Newton Solver

The solver starts at $b = 0$, which is feasible because $0$ is the lowest bound. Each iteration performs these steps:

1. Compute the gradient $g = Hb - a$.
2. Stop if the projected step is negligible: $\max_j \left|\operatorname{clip}(b_j - g_j,\, 0,\, y_j) - b_j\right| \le \epsilon$, where $\epsilon = \max(10^{-10},\, 10^{-13}\max_j H_{jj})$. The second term keeps the test above gradient roundoff when stiffness is large.
3. Hold fixed any variable sitting on a bound whose gradient pushes outward (at $0$ with $g \ge 0$, or at $y_j$ with $g \le 0$). Solve the Newton system for the remaining free variables. A tiny ridge keeps this sparse solve stable.
4. Backtrack along the step: clip to the bounds, then halve the step until the Armijo condition $E(b^+) \le E(b) + 10^{-4}\,g^\top(b^+ - b)$ holds. If the step falls below $10^{-12}$, the search ends.

Because the energy is quadratic, an unconstrained Newton step would be exact. The bounds make the method iterative, and the solver allows up to 1000 iterations.

#### 5. Convergence

If the iteration limit is reached or the line search stalls before the stopping test passes, the fit raises `RuntimeError("Cloth baseline fit did not converge")`. The optimizer never returns an unconverged baseline. Batch runs record this as a failure for that file.

#### 6. Effect of Stiffness

- Very small $\lambda$: the curvature penalty is negligible, so the cloth rises to the ceiling at almost every knot and follows narrow structure closely.
- Moderate $\lambda$: the cloth bends to follow broad background changes while staying under the signal.
- Large $\lambda$: curvature is nearly zero, so the baseline approaches a straight line beneath the signal and may leave a larger gap under broad structure.

The slider changes $\lambda$ on a logarithmic scale because useful stiffness values span several orders of magnitude. The default is `9e-5`.

## Options

| Option | Description |
| --- | --- |
| `--stiffness VALUE` | Cloth curvature resistance; default is `9e-5`. Lower values allow more bending; higher values produce a smoother baseline. |
| `--baseline-points COUNT` | Number of coarse baseline points; default is `226`. |
| `--signal-type TYPE` | `auto`, `absorbance`, or `transmittance`; default is `auto`. |
| `--display-as UNITS` | Plot in `source`, `absorbance`, or `transmittance` units; default is `source`. Conversion affects the plot only. |
| `--fit-range MIN MAX` | Restrict baseline fitting to this wavenumber range. Repeat to specify multiple ranges. |
| `--ignore-range MIN MAX` | Exclude this range from fitting and smoothly bridge the baseline across it. Repeat for multiple ranges. |
| `--no-plot` | Disable plotting. Plotting is on by default. |
| `--no-show-baseline-corrected` | Hide the baseline-corrected panel while keeping the main plot. It is shown by default. |
| `--output PATH` | Set the corrected-spectrum output path for a single input. |
| `--output-dir PATH` | Set the directory for outputs from a batch run. |
| `--output-format FORMAT` | Choose `asp` or `csv`; default is `asp`. |
| `--output-units UNITS` | Save the corrected spectrum in `source`, `absorbance`, or `transmittance` units; default is `source`. Transmittance-to-absorbance conversion requires positive values. |
| `-h`, `--help` | Show all command-line options. |

Examples:

```powershell
# Save CSV output without opening the plot
python main.py samples/ABS-1.csv --no-plot --output-format csv

# Plot as absorbance, even when the source file contains transmittance
python main.py samples/ABS-1.csv --display-as absorbance

# Save the corrected spectrum as transmittance, regardless of the source units
python main.py samples/ABS-1.csv --no-plot --output-units transmittance --output-format csv

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
