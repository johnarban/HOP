import argparse
import warnings
import os
import numpy as np
from scipy import ndimage
import astropy.stats as astats
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import colors as mc
from astropy.modeling.fitting import SimplexLSQFitter
from astropy.modeling.models import Gaussian1D, Chebyshev1D

import helper_funcs as ju

hg = np.loadtxt('hgar_blue.txt') / 10
ar = np.loadtxt('argon_red.txt') / 10 


# def find_peaks(arr,threshold = 0.1 , size=20,axis=-1):
#     arr = ju.scale_ptp(arr)
#     maxarr = ndimage.maximum_filter1d(arr,size=size,axis=axis)
#     return np.where((arr == maxarr) & (arr > threshold))[0]


def get_cosmic_rays(data):
    x = ndimage.uniform_filter1d(data,5,axis=0)
    x2= ndimage.uniform_filter1d(data**2,5,axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cr = np.sqrt(x2 - x**2)
        ma = astats.sigma_clip(cr,sigma_lower=None,sigma_upper=10,)
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


def shift_row_interp(A, wave_poly,plot=True,fig=None,axs=None):
    """
    Shift rows in a spectra so that spectral
    lines are vertical. Aligns spectra with location
    of spectra on top row (row 0)
    
    wave_poly should either be the slope
    of an individual spectral feature
    
    or the output of np.polyfit for that
    spectral feature
    
    """
    if not hasattr(wave_poly, "__iter__"):
        wave_poly = [wave_poly, 0]
    else:
        wave_poly = list(wave_poly)
        wave_poly[-1] = 0
    ## Shift indices assuming a slope
    ## this normalizes to the 0th row of the array
    B = np.zeros_like(A)
    # get indices for array
    
    # get array of true coords
    x_image = np.arange(A.shape[1])
    # loop over rows
    for i in range(A.shape[0]):
        x_rect = x_image - np.polyval(wave_poly,i) # true coord x_0 = x_orig + i * dλ/dpix
        # We want to interpolate 
        y_rect = np.interp(x_image,x_rect,A[i,:])#,left=np.median(A),right=np.median(A))
        B[i,:] = y_rect
        
        
    if plot:
        if axs is None:
            if fig is None:
                fig,axs = plt.subplots(1,2,figsize=(12,4))
        vmin, vmax  = np.percentile(A,[50,99])
        norm = mc.SymLogNorm(100,vmin=vmin,vmax=vmax)

        ax = axs[0]
        if ax is not None:
            ax.imshow(A,aspect='auto',norm=norm,cmap='viridis')
            ax.set_title('Original')

        ax = axs[1]
        if ax is not None:
            ax.imshow(B,aspect='auto',norm=norm,cmap='viridis')
            ax.set_title('Shift by line slope')
    
    
    return B



def wavelength_cal(peaks,hg,ar,order=1,order2=1,return_match=True):
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
        return (800,.5)
    
    pix = np.append(pix,new_pix)
    wave = np.append(wave,new_wave)
    p = np.polyfit(pix,wave,order)
    
    linelist = np.append(hg,ar)
    new = λpeaks = np.polyval(p,peaks)
    c = linelist[np.argmin(np.abs(new - linelist[:,np.newaxis]),axis=0)]
    p = np.polyfit(peaks,c,order2)
    print('Wavelength solution')
    print(f'{p[0]:0.5g} x + {p[1]:0.5g}')
    if return_match:
        return p, c
    else:
        return p

def specextract(data,bottom=None,top=None,plot=True,fig=None,ax=None):

    # Get the vertical slice where the data is
    x = np.mean(data,axis=1)
    y = np.arange(data.shape[0])
    
    #if sl is None:
    #fitter = fitting.LevMarLSQFitter()
    fitthis = False
    if (top is None) or (bottom is None):
        fitthis = True
    if fitthis:
        fitter = SimplexLSQFitter()
        model = Gaussian1D(np.max(x),np.argmax(x),1) + Chebyshev1D(5)#models.Const1D(0)
        fit = fitter(model,y,x,maxiter=10000)
        data = data# - fit.amplitude_1
        mean = fit.mean_0
        fwhm = fit.stddev_0 * 2.3548
        new_top,new_bot = int(mean + 2*fwhm),int(mean - 2*fwhm)
        if top is None:
            top = new_top
        if bottom is None:
            bottom = new_bot
    
    if bottom < top:
        sl = slice(max([0,bottom]),min([top,y.max()]))
    else:
        sl = slice(min([0,bottom]),min([top,y.max()]))
    print('Boundaries for spectra',sl.start,sl.stop)
    

    if plot:
        if ax is None:
            if fig is None:
                fig, ax = plt.subplots(1,1)
            else:
                ax = fig.add_subplot()    
        ax.plot(y,x,label='data')
        if fitthis:
            ax.plot(y,fit(y),'r',label='fit')
        ax.set_xlabel('cross-dispersion direction (pixels)')
        ax.set_ylabel('data units')
        ax.axvspan(sl.start,sl.stop,color='0.5')
        ax.legend(loc='best')
    
    return sl

def rectify_ccd(cal, order=1, plot=True,fig=None,ax=None):
    """ find the slope of spectral lines
    in the image plane of the CCD"""
    x_line = np.argmax(cal,axis=1)
    y = np.arange(cal.shape[0])
    
    # only fit 90% of the line to exclude noise
    clean = cal[y,x_line] > np.percentile(cal[y,x_line],50)
    # derive the shift
    p_disp = np.polyfit(y[clean],x_line[clean],order)
    # p0,dxdy,x_0 = p_disp

    
    if plot:
        vmin, vmax  = np.percentile(cal,[50,99])
        norm = mc.SymLogNorm(100,vmin=vmin,vmax=vmax)
        if ax is None:
            if fig is None:
                fig, ax = plt.subplots(1,1)
            else:
                ax = fig.add_subplot()    

        ax.imshow(cal,aspect='auto',norm=norm,cmap='magma')
        ax.plot(x_line[clean],y[clean],'b',lw=3)
        ax.plot(np.polyval(p_disp,y),y,'w--')
        ax.set_title('Original')

    # x_{line,0} = x_{line} - \frac{dx}{dy} * y
    y, x_line  = np.indices(cal.shape)
    p_disp[-1]=0 
    full_frame_solution = x_line - np.polyval(p_disp,y)
    return p_disp, full_frame_solution

def get_wavelength_solution(cal,threshold=0.05,size=5,plot=False):
    """
    returns the wavelength solution
    for 
    returns p, λ, rect
    
    
    """
    cal_spec = np.nanmean(cal,axis=0)
    peaks = ju.find_peaks(cal_spec,threshold=threshold,size=size,)[::-1] # put in wavelength order
    p = wavelength_cal(peaks,hg,ar)
    x = np.arange(len(cal_spec))
    λ = np.polyval(p,x)
    return p, λ, cal_spec



def reduce(cal_file, data_file, cal_threshold=0.05, bottom = None, top = None, rectify=False,
           plot=True,clip_cal=False,cosmic_rays=False,save=False):
    cal = fits.getdata(cal_file).astype(float)
    cal = cal - np.median(cal)
    obj = fits.getheader(data_file)['OBJECT']
    data = fits.getdata(data_file).astype(float)
    with plt.rc_context(rc={'image.origin': 'upper'}):
        if plot:
            mosaic = """AABB;CCDD;.II.;EEFF;GGHH"""
            fig, ax = plt.subplot_mosaic(mosaic,gridspec_kw={'height_ratios':[1,1,.7,1,1.25]},figsize=(12,16))
        else:
            mosaic = """GH"""
            fig, ax = plt.subplot_mosaic(mosaic,figsize=(10,4))
            ax['A']=None
            ax['B']=None
            ax['C']=None
            ax['D']=None
            ax['E']=None
            ax['F']=None
            ax['I']=None

        rect_sol, full_frame_solution = rectify_ccd(cal,plot=plot,ax=ax['A']);
        if rectify:
            cal = shift_row_interp(cal,rect_sol,plot=plot,axs=(None,ax['B']))
            data = shift_row_interp(data,rect_sol,plot=plot,axs=(ax['C'],ax['D']))
        else:
            shift_row_interp(cal,rect_sol,plot=plot,axs=(None,None))
            shift_row_interp(data,rect_sol,plot=plot,axs=(ax['C'],None))

        data = data - np.median(data)

        if clip_cal:
            trim = clip_calib(cal)

        sl = specextract(data,bottom=bottom,top=top,plot=plot,ax=ax['I'])

        # plot the section we will extract
        if plot:
            vmin, vmax  = np.percentile(cal,[50,99])
            norm = mc.SymLogNorm(100,vmin=vmin,vmax=vmax)
            plt.sca(ax['E'])
            plt.imshow(cal,norm=norm,cmap='viridis')
            plt.axhline(sl.start,color='r')
            plt.axhline(sl.stop,color='r')
            plt.title('Calibration')
            plt.sca(ax['F'])
            vmin, vmax  = np.percentile(data,[50,99])
            norm = mc.SymLogNorm(100,vmin=vmin,vmax=vmax)
            plt.imshow(data,norm=norm,cmap='viridis')
            plt.axhline(sl.start,color='r')
            plt.axhline(sl.stop,color='r')
            plt.colorbar()
            plt.title('Target')

        # Get wavelength calibration
        cal_spec = np.nanmean(cal[sl,:],axis=0)
        back = ju.findback1d(cal_spec,s=20)
        peaks = ju.find_peaks(cal_spec - back,threshold=.05,size=5,)[::-1] # put in wavelength order

        a = ax['G']
        p,matched = wavelength_cal(peaks,hg,ar,return_match=True)
        wavesol = np.polyval(p,np.arange(len(cal_spec)))

        #Plot wavelength cal
        a.plot(wavesol,cal_spec,lw=1)
        x = np.arange(len(cal_spec))
        a.plot(np.interp(peaks,x,wavesol),np.interp(peaks,x,cal_spec),'r.')
        a.plot(wavesol,back,lw=1)
        for i in matched:
            a.axvline(i,color='0.5',zorder=0)
        a.invert_xaxis()
        a.set_title('wavelength cal')
        a.set_xlabel('Wavelength [nm]')

        if cosmic_rays:
            cr = get_cosmic_rays(data)
            im2 = np.zeros(cr.shape + (4,))
            im2[:, :, 0] = 1
            im2[:, :, 3] = cr * 1
            if ax['F'] is not None:
                ax['F'].imshow(im2)
            spec = np.ma.mean(np.ma.masked_array(data,cr)[sl,:],axis=0)
        else:
            spec = np.mean(data[sl,:],axis=0)
        a = ax['H']
        
        #noise = np.nanstd(data[sl,:],axis=0)/np.sqrt(top-bot)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            noise = astats.sigma_clipped_stats(data,sigma=3,)[-1] / np.sqrt(sl.stop-sl.start)
            noise = np.sqrt(spec + noise**2)
        ju.errorbar_fill(wavesol,spec,noise,mid=True,alpha=1,lw=.5,ax=a)
        #plt.plot(wavesol,data[sl,:].T,'k',lw=.5)

        a.set_xlabel('Wavelength [nm]')
        a.set_ylabel('Uncalibrated data units')
        a.set_title(f'Spectrum of {obj}')
        
        out = list(zip(wavesol,spec,noise))
        fname = f"{data_file.replace('.FIT','.tsv')}"
        
        if save:
            fname = f"{data_file.replace('.FIT','.tsv')}"
            with open(fname,'w') as f:
                f.write('wavelength\tspectrum\terror\n')
                for i in out:
                    f.write('{:<9.3f}\t{:>10.3f}\t{:>10.3f}\n'.format(*i))
            #np.savetxt(fname,out,fmt='%16.5f',delimiter='\t',header='wavelength spectrum error')

        return (wavesol,spec,noise),fig
    
    



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Reduce a spectrum from the spectrograph on the Clay Telescope (Harvard University)")

    parser.add_argument("-c","--cal",help="""The calibration file""")

    parser.add_argument("-d","--data",help="The spectrum we want reduced")
    
    parser.add_argument("-b","--bottom",default = None,type=int,help="lower index for spectrum boundary")
    
    parser.add_argument("-t","--top",default = None,type=int,help="higher index for spectrum boundary")

    parser.add_argument("--threshold",default = 0.05,type=float,help="Threshold level used for defining getting peaks for calibration")
    
    parser.add_argument("--plot",action='store_true',help="Plot only the spectrum")
    
    parser.add_argument("-r","--dont-rectify",action='store_false',help="Don't rectify the spectra?")
    
    parser.add_argument("-s","--dont-save",action='store_false',help="Don't Rectify the spectra?")
    
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
    
    rectify = args.dont_rectify
    save = args.dont_save
    
    zap_cosmic_rays = args.zap_cosmic_rays
    
    plot = not args.plot
    
    hg = np.loadtxt('hgar_blue.txt') / 10
    ar = np.loadtxt('argon_red.txt') / 10 

    out,fig = reduce(cal_file, data_file, cal_threshold=threshold, bottom = bottom, top = top, rectify=rectify,cosmic_rays=zap_cosmic_rays,plot=plot,save=save)
    
    
    fname = f"{data_file.replace('.FIT','_big.png')}"
    fig.savefig(fname)
    
    if not args.batch:
        os.system(f'open {fname}')