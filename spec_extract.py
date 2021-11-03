# import needed packages
import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling.fitting import SimplexLSQFitter, LevMarLSQFitter
from astropy.modeling.models import Gaussian1D, Chebyshev1D





# fit a gaussian and baseline to 1D spectra
def fit_gauss(x, y, plot=False, fig=None, ax=None):
    """Fit a gaussian to a 1D spectrum
    Parameters
    ----------
    x : array
        x-axis of the spectrum
    y : array
        y-axis of the spectrum
    plot [False] : bool
        if True, plot the spectrum and the fit
    fig [None] : matplotlib figure
        figure to plot on
    ax [None] : matplotlib axis
        axis to plot on
    Returns
    -------
    model : astropy model
        the fitted model
    """
    if plot:
        if ax is None:
            if fig is None:
                fig, ax = plt.subplots(1, 1)
            else:
                ax = fig.add_subplot()
    fitter = LevMarLSQFitter()
    line = Gaussian1D(amplitude=np.max(y), mean=np.argmax(y), stddev=5.)
    model = line + Chebyshev1D(5)
    fit = fitter(model, x, y, maxiter=10000)

    if plot:
        ax.plot(x, y, label='data')
        ax.plot(x, fit(x), 'r', label='fit')
        ax.set_xlabel('cross-dispersion direction (pixels)')
        ax.set_ylabel('data units')
        ax.legend(loc='best')

    hwhm = fit.stddev_0 * 2.3548 / 2 # HWHM
    mean = fit.mean_0
    return mean, hwhm, fit


def specextract(data, bottom=None, top=None, slice_fwhm=1,
                plot=False, fig=None, ax=None, aperture=False,sbig=False):
    """Extract spectra
    """
    # Get the vertical slice where the data is
    x = np.nanmean(data, axis=1)
    y = np.arange(data.shape[0])

    fitthis = False
    if (top is None) or (bottom is None):
        fitthis = True

    if sbig:
        center = np.argmax(x)
        top = center + 10
        bottom = center - 10
        fitthis = False
    elif fitthis:
        print('Fitting')
        mean, hwhm, fit = fit_gauss(y, x) # fit a gaussian to the data
        new_top, new_bot = (int(mean + slice_fwhm*hwhm),
                            int(mean - slice_fwhm*hwhm))

        top = top or new_top
        bottom = bottom or new_bot

    # sort top and bottom
    indices = [top,bottom]
    indices.sort()
    bottom, top = indices
    # bottom should be lowest index
    if bottom < 0:
        bottom = 0
    # top should be highest index
    if top > y.max():
        top = y.max()

    sl = slice(bottom, top)
    print('Boundaries for spectra', sl.start, sl.stop)

    if plot:
        if ax is None:
            if fig is None:
                fig, ax = plt.subplots(1, 1)
            else:
                ax = fig.add_subplot()
        ax.plot(y, x, label='data')
        if fitthis:
            ax.plot(y, fit(y), 'r', label='fit')
        ax.set_xlabel('cross-dispersion direction (pixels)')
        ax.set_ylabel('data units')
        ax.axvspan(sl.start, sl.stop, color='0.5')
        ax.legend(loc='best')

    return sl
