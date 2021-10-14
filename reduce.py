import argparse
import warnings
import numpy as np
from scipy import ndimage
import astropy.stats as astats
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.modeling.fitting import SimplexLSQFitter
from astropy.modeling.models import Gaussian1D, Chebyshev1D

from helper_funcs import *

# Data reduction steps
##     > cal = fits.getdata(cal_file)
##     > data = fits.getdata(data_file)
## 1) Optionally rectify the spectra: 
##     > spatmap = rectify_ccd(cal)[0]
##     > cal = shift_row_interp(cal,spatmap)
##     > data = shift_row_interp(data,spatmap)
## 2) Get rows with the spectra
##    This can be done manually or automatically
##     > # Manual
##     > sl = specextract(data, bottom=value, top=value)
##     > # Automatic
##     > sl = specextract(data) 
## 2) Get Wavelength calibration
##     > wavelengths, cal_spec, p = get_wavelength_cal(cal[sl,:])
## 3) Get spectrum
##     > spectrum = np.mean(data[sl,:],axis=0)
## 4) Optionally get noise
##     > Nrow = sl.stop-sl.start
##     > noise = astats.sigma_clipped_stats(data,sigma=3,)[-1] / np.sqrt(Nrow)
##     > noise = np.sqrt(spec + noise**2)
##
## For convenience, all this is wrapped in a single function `reduce`
##  > reduce(cal_file, data_file)

def get_cosmic_rays(data,n=5,sigma_upper=10):
    """
    Simple method of removing cosmic rays
    ~based on 2004PASP..116..148P (Pych 2004)
    
    1) Make map of local standard deviation
    2) sigma-clip values with very high stdev
    """
    x = ndimage.uniform_filter1d(data,n,axis=0)
    x2= ndimage.uniform_filter1d(data**2,n,axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    cr = np.sqrt(x2 -x**2)
    ma = astats.sigma_clip(cr,sigma_lower=None,sigma_upper=sigma_upper,)
    cr = ma.mask
    return cr


def rectify_ccd(cal, order=1, per_lim = 30):
    """ find the slope of spectral lines
    in the image plane of the CCD
    """
    
    # Get the cross-dispersion profile of brightest line
    x_line = np.argmax(cal,axis=1)
    y = np.arange(cal.shape[0])
    ## To prevent contamination, only use a fraction of the line
    clean = cal[y,x_line] > np.percentile(cal[y,x_line],per_lim)
    
    # Fit cross-dispersion profile
    p_disp = np.polyfit(y[clean],x_line[clean],order)
    p_disp[-1] = 0 # remove the offset (x_line_0) specific to the brightest line
    
    
    ## we'll never use this, but nice to have
    ## create array of rectified axes cooordinate
    y, x_lines  = np.indices(cal.shape) # returns m, b ... x_line = m y + x_line_0
    full_frame_solution = x_lines - np.polyval(p_disp,y)
    
    return p_disp , full_frame_solution


def shift_row_interp(A, wave_poly):
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
    
    ## original non-rectified coords
    x_image = np.arange(A.shape[1]) 
    for i in range(A.shape[0]):
        # get rectified pixel coordinates
        x_rect = x_image - np.polyval(wave_poly,i) 
        
        # interpolate values to fill array
        y_rect = np.interp(x_image,x_rect,A[i,:],left=np.median(A),right=np.median(A))
        B[i,:] = y_rect
    
    return B


def wavelength_cal(peaks,hg,ar):
    try:
        # Mercury is easy to calibrate
        pix = peaks[:3] # 0,1,2
        wave = hg[-3:]
        # wavelength solution linear
        # Argon 
        # Want to just add the first peak #strongest peak
        new_pix = peaks[3] #peaks[3:][np.argmax(cal_spec[peaks[3:]])]
        new_wave = ar[0]#ar[6]
    except:
        print('Failed to find peaks')
        print('What are the pixel coords of the hilighted peaks?')
        plt.figure()
        plt.plot(hg,hg/hg,'o',color='dodgerblue',label='Hg',mec='k',mew=1)
        plt.plot(ar,ar/ar,'ro',label='Ar',mec='k',mew=1)
        for i in np.append(hg[-3:],ar[0]):
            plt.axvline(i,color='k',zorder=0,lw=.5)
        plt.title(r'Location of Hg and Ar lines')
        plt.xlabel('Wavelength [nm]')
        plt.legend()
        

        peaks = input('peaks (separate with space): ')
        peaks = [int(i) for i in peaks.split()]
        plt.close()
        wavelength_cal(peaks,hg,ar)
    
    pix = np.append(pix,new_pix)
    wave = np.append(wave,new_wave)
    p = np.polyfit(pix,wave,1)
    print('Wavelength solution')
    print(f'{p[0]:0.5g} x + {p[1]:0.5g}')
    
    return p

def get_wavelength_cal(cal,threshold=0.05,size=5):
    """ returns wavelength solution
        from a prepared calibration image
    """    
    cal_spec = np.nanmean(cal,axis=0)
    peaks = find_peaks(cal_spec,threshold=threshold,size=size,)[::-1] # put in wavelength order
    p = wavelength_cal(peaks,hg,ar)
    λ = np.polyval(p,np.arange(cal.shape[1]))
    return λ, cal_spec, p

def get_wavelength_solution(cal,threshold=0.05,size=5):
    """ returns wavelength solution
        for calibration image. if only a file
        is given, then rectify
    """
    if isinstance(cal,str):
        cal = fits.getdata(cal)
        spatmap = rectify_ccd(cal)
        cal = shift_row_interp(cal,spatmap)
    
    cal_spec = np.nanmean(cal,axis=0)
    peaks = find_peaks(cal_spec,threshold=threshold,size=size,)[::-1] # put in wavelength order
    p = wavelength_cal(peaks,hg,ar)
    λ = np.polyval(p,np.arange(cal.shape[1]))
    return p, λ, cal

def specextract(data,bottom=None,top=None):

    # Get the vertical slice where the data is
    x = np.mean(data,axis=1)
    y = np.arange(data.shape[0])
    
    #if sl is None:
    #fitter = fitting.LevMarLSQFitter()
    fitter = SimplexLSQFitter()
    model = Gaussian1D(np.max(x),np.argmax(x),1) + Chebyshev1D(5)#models.Const1D(0)
    fit = fitter(model,y,x,maxiter=10000)
    data = data
    mean = fit.mean_0
    fwhm = fit.stddev_0 * 2.3548
    new_top,new_bot = int(mean + 2*fwhm),int(mean - 2*fwhm)
    if top is None:
        top = new_top
    if bottom is None:
        bottom = new_bot
    
    sl = slice(max([0,bottom]),min([top,y.max()]))
    
    return sl



def reduce(cal_file, data_file, bias=None, cal_threshold=0.05, bottom = None, top = None, rectify=False,
           plot=False,clip_cal=False,cosmic_rays=False,save=False):
    cal = fits.getdata(cal_file).astype(float)
    cal = cal - np.median(cal)
    obj = fits.getheader(data_file)['OBJECT']
    data = fits.getdata(data_file).astype(float)
    
    # 1) Rectify data
    rect_sol, full_frame_solution = rectify_ccd(cal);
    if rectify:
        cal = shift_row_interp(cal,rect_sol)
        data = shift_row_interp(data,rect_sol)
    
    # remove offset/bias
    data = data - np.median(data)
    
    # 2) Extract selection for spectral regions
    sl = specextract(data,bottom=bottom,top=top)
    
    # 3) Get wavelength calibration
    wavesol, cal_spec, p = get_wavelength_cal(cal[sl,:])
    
    
    # 4) Some targets are cosmic rays (this is very simplistic)
    if cosmic_rays:
        cr = get_cosmic_rays(data)
        spec = np.ma.mean(np.ma.masked_array(data,cr)[sl,:],axis=0)
    else:
        spec = np.mean(data[sl,:],axis=0)


    # 5) Measure the noise
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        noise = astats.sigma_clipped_stats(data,sigma=3,)[-1] / np.sqrt(sl.stop-sl.start)
        noise = np.sqrt(spec + noise**2)
   
    out = list(zip(wavesol,spec,noise))
    if save:
        fname = f"{data_file.replace('.FIT','.tsv')}"
        with open(fname,'w') as f:
            f.write('wavelength\tspectrum\terror\n')
            for i in out:
                f.write('{:<9.3f}\t{:>10.3f}\t{:>10.3f}\n'.format(*i))
                    
    return wavesol,spec,noise




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Reduce a spectrum from the spectrograph on the Clay Telescope (Harvard University)")

    parser.add_argument("-c","--cal",help="""The calibration file""")

    parser.add_argument("-d","--data",help="The spectrum we want reduced")
    
    parser.add_argument("-b","--bottom",default = None,type=int,help="lower index for spectrum boundary")
    
    parser.add_argument("-t","--top",default = None,type=int,help="higher index for spectrum boundary")

    parser.add_argument("--threshold",default = 0.05,type=float,help="Threshold level used for defining getting peaks for calibration")
    
    parser.add_argument("-p","--plot",action='store_false',help="don't plot the spectrum")
    
    parser.add_argument("-r","--rectify",action='store_false',help="don't rectify the spectra?")
    
    parser.add_argument("-s","--dont-save",action='store_true',help="Don't save the spectra?")
    
    parser.add_argument("--batch",action='store_true',help="Don't open images")
    
    parser.add_argument("--zap-cosmic-rays",default=False,help=""""Get rid of cosmic rays. 
                                                        This uses simple algorithm that could clip bright lines. 
                                                        Best used only for long integrations of diffuse nebular emission""")
    
    args = parser.parse_args()
    
    cal_file = args.cal
    data_file = args.data
    
    threshold = args.threshold
    
    bottom = args.bottom
    top = args.top
    
    rectify = args.rectify
    
    zap_cosmic_rays = args.zap_cosmic_rays
    
    plot = args.plot
    
    save = not args.dont_save
    
    hg = np.loadtxt('hgar_blue.txt') / 10
    ar = np.loadtxt('argon_red.txt') / 10 

    out = reduce(cal_file, data_file, cal_threshold=threshold, bottom = bottom, top = top, rectify=rectify,cosmic_rays=zap_cosmic_rays,save=save)
    
    
    if plot:
        w,spec,noise = out
        plt.figure(facecolor='w')
        plt.fill_between(w,spec-noise,spec+noise)
        plt.plot(w,spec)
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('arbitraty data units')
        fname = f"{data_file.replace('.FIT','.png')}"
        plt.savefig(fname)
        if not args.batch:
            plt.show(block=True)
        
        