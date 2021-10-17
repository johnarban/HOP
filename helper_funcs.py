# Some non-spectroscopy specific functions
from skimage.segmentation import expand_labels
from astropy.convolution import kernels
from astropy.convolution import convolve_fft, convolve
import astropy.stats as astats
import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import matplotlib.colors as mc
from colorsys import rgb_to_hls, hls_to_rgb
from astropy.modeling.fitting import SimplexLSQFitter
from astropy.modeling.models import Gaussian1D, Chebyshev1D

import wavelength_cal as wc


def getheader_val(fname, card_name):
    """
    Pure python function to get header values
    from a fits file

    """
    with open(fname, 'rb') as f:
        line = f.read(80)
        while line.strip() != b'END':
            if line[:7] != b'HISTORY':
                card_value = line.split(b'/')[0]
                card, value = card_value.split(b'=')
                if card.strip().decode() == card_name:
                    return value.strip().decode().replace("'", "")
            line = f.read(80)


def adjust_lightness(color, amount=0.5):
    # brighter : amount > 1
    # https://stackoverflow.com/a/49601444
    try:
        c = mc.cnames[color]
    except Exception:
        c = color
    c = rgb_to_hls(*mc.to_rgb(c))
    return hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def errorbar_fill(x=None, y=None, yerr=None, *args,
                  ax=None, mid=True, color=None, alpha=1, lw=1, ls="-",
                  fmt=None, label=None, **kwargs,):
    if ax is None:
        ax = plt.gca()

    if mid:
        alpha_fill = alpha * 2
        if alpha_fill >= 1:
            alpha_fill = 1
    if color is None:
        color = ax.plot([], [])[0].get_color()
    ax.fill_between(x, y - yerr, y + yerr, color=adjust_lightness(color,
                    1.75), alpha=alpha*.75, label=label, **kwargs)
    if mid:
        ax.plot(x, y, color=color, alpha=alpha, lw=lw, ls=ls, **kwargs)

    return ax


def peak_widths(cal_spec, peaks, size=5):
    pix = np.zeros(cal_spec.shape)
    pix[peaks] = peaks
    h = np.diff(peaks).min()//2
    for i in range(min(h, size)):
        pix = expand_labels(pix, 1)
    return pix


def scale_ptp(arr):
    """
    scale an array to between 0 and 1
    (arr - min)/(max-min)
    """
    g = np.isfinite(arr)
    if g.any():
        return (arr - np.nanmin(arr[g]))/np.ptp(arr[g])
    else:
        return arr

# from scipy.signal import peak_widths


def find_centroid(cal_spec, peaks, size=5):
    def centroid(x, y):
        # return np.sum(x*y)/np.sum(y)
        a, b, c = np.polyfit(x, y, 2)
        return -b/(2*a), -b**2/(4*a) + c

    # widths,height,left,right = peak_widths(cal_spec,peaks,.75)
    pix = peak_widths(cal_spec, peaks, size=size)
    centroids = []
    for i in range(len(peaks)):
        mask = pix == peaks[i]
        # w = widths[i]
        # p = peaks[i]
        # h = height[i]
        # l = left[i]
        # r = right[i]

        x = np.arange(len(cal_spec))
        y = cal_spec

        x_cen = x[mask]  # np.hstack([l,x[(x>l) & (x<r)],r])
        y_cen = y[mask]  # np.hstack([h,y[(x>l) & (x<r)],h])

#         x_int = np.linspace(l,r,20)
#         y_int = np.interp(x_int,x,y)
        centroids.append(centroid(x_cen, y_cen))
    return np.array(centroids).T


def find_peaks(arr, threshold=0.05, size=5, axis=-1, centroid=True):
    """
    Scales array so min(arr)=0, max(arr)=1
    thresold: minimum scaled value for peak
        default: 0.05
    size: width of peak like features
        default: 5

    returns: indices of peaks
    """
    arr_scale = scale_ptp(arr)
    maxarr = ndimage.maximum_filter1d(arr_scale, size=size, axis=axis)
    peaks = np.where((arr_scale == maxarr) & (arr_scale > threshold))[0]
    if centroid:
        cen = find_centroid(arr, peaks, size=size)
        return cen
    else:
        return peaks, arr[peaks]


def findback1d(image, s=31, fill=0, experimental=False):
    image = np.copy(image)
    oldnan = np.isnan(image)

    sp = s + s//4
    sp = sp+1 if sp % 2 == 0 else sp
    s1 = sp//2
    s1 = s1+1 if s1 % 2 == 0 else s1
    s2 = s1//2
    arr = np.ones((sp,))
    arr[s1-s2:s1+s2] = 0
    expand = convolve_fft(image, kernels.CustomKernel(arr), boundary='wrap')
    image[np.isnan(image)] = expand[np.isnan(image)]

    # image[oldnan] = fill
    s = int(s)
    s = s+1 if s % 2 == 0 else s
    bkg_min = ndimage.minimum_filter(image, size=(s,))
    bkg_max = ndimage.maximum_filter(bkg_min, size=(s,))
    kernel = kernels.Box1DKernel(s)
    bkg_mean = convolve(bkg_max, kernel, boundary='extend',)

    if experimental:
        bkg_mean = np.min(np.vstack([[bkg_max], [bkg_mean]]), axis=0)
        bkg_new = np.copy(bkg_mean)
        # print(bkg_new.shape)
        s = s//2
        while s > 2:
            s = s+1 if s % 2 == 0 else s
            kernel2 = kernels.Box1DKernel(s)
            bkg_mean = convolve_fft(bkg_new, kernel2, boundary='wrap')
            bkg_new = np.min(np.array([bkg_mean, bkg_new]), axis=0)
            s = s//2

        bkg_new = np.min(np.vstack([[bkg_mean], [bkg_new]]), axis=0)
        kernel3 = kernels.CustomKernel(np.ones((1,)))
        kernel3.normalize()
        bkg_mean = convolve_fft(bkg_new, kernel, boundary='wrap')

    bkg = bkg_mean

    bkg[oldnan] = np.nan

    return bkg


# def find_peaks(arr,threshold = 0.1 , size=20,axis=-1):
#     arr = ju.scale_ptp(arr)
#     maxarr = ndimage.maximum_filter1d(arr,size=size,axis=axis)
#     return np.where((arr == maxarr) & (arr > threshold))[0]


def get_cosmic_rays(data):
    x = ndimage.uniform_filter1d(data, 5, axis=0)
    x2 = ndimage.uniform_filter1d(data**2, 5, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cr = np.sqrt(x2 - x**2)
        ma = astats.sigma_clip(cr, sigma_lower=None, sigma_upper=10,)
    cr = ma.mask
    return cr


def shift_rows(A, r):
    # https://stackoverflow.com/a/20361561
    rows, column_indices = np.ogrid[:A.shape[0], :A.shape[1]]

    # Use always a negative shift, so that column_indices are valid.
    # (could also use module operation)
    r[r < 0] += A.shape[1]
    column_indices = column_indices - r[:, np.newaxis]

    return A[rows, column_indices]


def shift_row_interp(A, wave_poly, plot=False, fig=None, axs=None):
    """
    Shift rows in a spectra so that spectral
    lines are vertical. Aligns spectra with location
    of spectra on top row (row 0)

    wave_poly should either be the slope
    of an individual spectral feature

    or the output of np.polyfit for that
    spectral feature

    """
    # Make sure wave_poly is in the
    # correct format
    if not hasattr(wave_poly, "__iter__"):
        wave_poly = [wave_poly, 0]
    else:

        wave_poly = list(wave_poly)
        wave_poly[-1] = 0

    # create empty array to fill
    # with rectified spectra
    B = np.zeros_like(A)

    # Loop to rectify array

    # original non-rectified coords
    x_image = np.arange(A.shape[1])
    for i in range(A.shape[0]):
        # true coord x_0 = x_orig + i * dλ/dpix
        x_rect = x_image - np.polyval(wave_poly, i)
        # We want to interpolate
        # ,left=np.median(A),right=np.median(A))
        y_rect = np.interp(x_image, x_rect, A[i, :])
        B[i, :] = y_rect

    if plot:
        if axs is None:
            if fig is None:
                fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        vmin, vmax = np.percentile(A, [50, 99])
        norm = mc.SymLogNorm(100, vmin=vmin, vmax=vmax)

        ax = axs[0]
        if ax is not None:
            ax.imshow(A, aspect='auto', norm=norm, cmap='viridis')
            ax.set_title('Original')

        ax = axs[1]
        if ax is not None:
            ax.imshow(B, aspect='auto', norm=norm, cmap='viridis')
            ax.set_title('Shift by line slope')

    return B


def wavelength_cal(peaks, hg, ar, order=1, order2=1, return_match=True):
    try:
        # Mercury is easy to calibrate
        pix = peaks[:3]  # 0,1,2
        wave = hg[-3:]
        # wavelength solution linear
        # Argon
        # Want to just add the first peak #strongest peak
        new_pix = peaks[3]  # peaks[3:][np.argmax(cal_spec[peaks[3:]])]
        new_wave = ar[0]  # ar[6]
    except Exception:
        print('Failed to find peaks')
        return (800, .5)

    pix = np.append(pix, new_pix)
    wave = np.append(wave, new_wave)
    p = np.polyfit(pix, wave, order)

    linelist = np.append(hg, ar)
    new = np.polyval(p, peaks)
    c = linelist[np.argmin(np.abs(new - linelist[:, np.newaxis]), axis=0)]
    p = np.polyfit(peaks, c, order2)
    print('Wavelength solution')
    print(f'{p[-2]:0.5g} x + {p[-1]:0.5g}')
    if return_match:
        return p, c
    else:
        return p


def clamp(x, xmin, xmax):
    # force xmin < x < xmax
    return max(xmin, min(x, xmax))


def specextract(data, bottom=None, top=None, slice_fwhm=1.5,
                plot=False, fig=None, ax=None):

    # Get the vertical slice where the data is
    x = np.mean(data, axis=1)
    y = np.arange(data.shape[0])

    # if sl is None:
    # fitter = fitting.LevMarLSQFitter()
    fitthis = False
    if (top is None) or (bottom is None):
        fitthis = True
    if fitthis:
        fitter = SimplexLSQFitter()
        model = Gaussian1D(np.max(x), np.argmax(x), 1) + \
            Chebyshev1D(5)  # models.Const1D(0)
        fit = fitter(model, y, x, maxiter=10000)
        data = data  # - fit.amplitude_1
        fwhm = fit.stddev_0 * 2.3548
        new_top, new_bot = (int((fit.mean_0) + slice_fwhm*fwhm),
                            int((fit.mean_0) - slice_fwhm*fwhm))
        if top is None:
            top = new_top
        if bottom is None:
            bottom = new_bot

    if bottom < top:
        if top < y.max():
            if bottom < 0:
                bottom = 0
        else:
            top = 0
    else:
        if top > 0:
            bottom = int(top//2)
        else:
            bottom, top = 0, y.max()
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


def rectify_ccd(cal, order=2, per_lim=5, plot=False, fig=None, ax=None):
    """ find the slope of spectral lines
    in the image plane of the CCD

    returns p_disp, full_frame_solution

    """
    x_line = np.argmax(cal, axis=1)
    y = np.arange(cal.shape[0])

    # only fit 90% of the line to exclude noise
    clean = cal[y, x_line] > np.percentile(cal[y, x_line], per_lim)
    # derive the shift
    p_disp = np.polyfit(y[clean], x_line[clean], order)
    # remove the offset (x_line_0) specific to the brightest line
    p_disp[-1] = 0

    if plot:
        vmin, vmax = np.percentile(cal, [50, 99])
        norm = mc.SymLogNorm(100, vmin=vmin, vmax=vmax)
        if ax is None:
            if fig is None:
                fig, ax = plt.subplots(1, 1)
            else:
                ax = fig.add_subplot()

        ax.imshow(cal, aspect='auto', norm=norm, cmap='magma')
        ax.plot(x_line[clean], y[clean], 'b', lw=3)
        ax.plot(np.polyval(p_disp, y), y, 'w--')
        ax.set_xlim(0,cal.shape[1])
        ax.set_title('Original')

    # we'll never use this, but nice to have
    # create array of rectified axes cooordinate
    y, x_line = np.indices(cal.shape)
    full_frame_solution = x_line - np.polyval(p_disp, y)

    return p_disp, full_frame_solution


def get_wavelength_cal(cal_file, recal='n', threshold=0.05, size=5, order=2):
    """
    # returns wavelength_solution, λ, calibration_spectrum, rectify_solution
    """
    return wc.wavelength_cal(cal_file, recal=recal,
                             order=order,
                             threshold=threshold,
                             size=size)
