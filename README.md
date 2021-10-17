# HOP


Required libraries

- `skimage`
- `numpy`
- `astropy`
- `scipy`
- `matplotlib`

These can all be installed using `pip install <name of package>`


To reduce a spectrum between pixels `(bottom, top)`

```bash
# if you want no plots
python reduce.py -c <calibration-file>  -d <data-file>  -t <top>  -b <bottom>
# if you want plots
python plot_and_reduce.py -c <calibration-file>  -d <data-file>  -t <top>  -b <bottom>

```
If you have not done spectral calibration for `<calibration-file>` it will open an interactive window where you can click the lines in the calibration spectrum and the corresponding lines from the Hg/Ar line list.

Some options for `reduce.py` and `plot_and_reduce.py` are

- `--recal`: force recalibration. Useful if you misidentified the line
- `--batch`: run without showing any plots. Useful if you create a script to reduce multiple spectra.
- `--plot`: [**Only in `plot_and_reduce.py`**] plot only the wavelength calibration and target spectrum without the diagnostic plots. Recommended if you using a script and `-t` and `-b` have been set.
- `--width`: How wide should the spectral extraction region be is measured automatically. Default is 1.5* gaussian FWHM.