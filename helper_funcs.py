# Some non-spectroscopy specific functions
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import matplotlib.colors as mc
from colorsys import rgb_to_hls,hls_to_rgb



def adjust_lightness(color, amount=0.5):
    # brighter : amount > 1
    #https://stackoverflow.com/a/49601444
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = rgb_to_hls(*mc.to_rgb(c))
    return hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def errorbar_fill(x=None,y=None,yerr=None,*args,
    ax=None,mid=True,color=None,alpha=1,lw=1,ls="-",
    fmt=None,label=None,**kwargs,):
    if ax is None:
        ax = plt.gca()


    if mid:
        alpha_fill = alpha * 2
        if alpha_fill >= 1:
            alpha_fill = 1
    if color is None:
        color = ax.plot([],[])[0].get_color()
    ax.fill_between(x, y - yerr, y + yerr, color=adjust_lightness(color,1.75), alpha=alpha*.75,label=label,**kwargs)
    if mid:
        ax.plot(x, y, color=color, alpha=alpha, lw=lw, ls=ls,**kwargs)

    return ax


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

from scipy.signal import peak_widths

def find_centroid(cal_spec, peaks):
    def centroid(x,y):
        return np.sum(x*y)/np.sum(y)
    

    widths,height,left,right = peak_widths(cal_spec,peaks)
    centroids = []
    for i in range(len(peaks)):
        w = widths[i]
        p = peaks[i]
        h = height[i]
        l = left[i]
        r = right[i]
        
        x = np.arange(len(cal_spec))
        y = cal_spec
        
        x_cen = np.hstack([l,x[(x>l) & (x<r)],r])
        y_cen = np.hstack([h,y[(x>l) & (x<r)],h])
        
#         x_int = np.linspace(l,r,20)
#         y_int = np.interp(x_int,x,y)
        centroids.append(centroid(x_cen,y_cen))
    return np.array(centroids)


def find_peaks(arr,threshold = 0.05 , size=5,axis=-1):
    """
    Scales array so min(arr)=0, max(arr)=1
    thresold: minimum scaled value for peak
        default: 0.05
    size: width of peak like features
        default: 5
    
    returns: indices of peaks
    """
    arr = scale_ptp(arr) 
    maxarr = ndimage.maximum_filter1d(arr,size=size,axis=axis)
    peaks = np.where((arr == maxarr) & (arr > threshold))[0]
    cen = find_centroid(arr,peaks)
    return cen



from astropy.convolution import convolve_fft, convolve
from astropy.convolution import kernels

def findback1d(image,s=31,fill=0,experimental=False):
    image = np.copy(image)
    oldnan = np.isnan(image)
    
    sp = s + s//4
    sp = sp+1 if sp%2==0 else sp
    s1 = sp//2
    s1 = s1+1 if s1%2==0 else s1
    s2 = s1//2
    arr=np.ones((sp,))
    arr[s1-s2:s1+s2]=0
    expand =convolve_fft(image,kernels.CustomKernel(arr),boundary='wrap')
    image[np.isnan(image)] = expand[np.isnan(image)]
    
    #image[oldnan] = fill
    s = int(s)
    s = s+1 if s%2==0 else s
    bkg_min = ndimage.minimum_filter(image,size=(s,))
    bkg_max = ndimage.maximum_filter(bkg_min,size=(s,))
    kernel = kernels.Box1DKernel(s)
    bkg_mean = convolve(bkg_max,kernel,boundary='extend',)
    
    
    if experimental:
        bkg_mean = np.min(np.vstack([[bkg_max],[bkg_mean]]),axis=0)
        bkg_new = np.copy(bkg_mean)
        #print(bkg_new.shape)
        s=s//2
        while s>2:
            s = s+1 if s%2==0 else s
            kernel2 = kernels.Box1DKernel(s)
            bkg_mean = convolve_fft(bkg_new,kernel2,boundary='wrap')
            bkg_new = np.min(np.array([bkg_mean,bkg_new]),axis=0)
            s=s//2

        bkg_new = np.min(np.vstack([[bkg_mean],[bkg_new]]),axis=0)
        kernel3 = kernels.CustomKernel(np.ones((1,)))
        kernel3.normalize()
        bkg_mean = convolve_fft(bkg_new,kernel,boundary='wrap')
    
    
    bkg = bkg_mean
    
    
    bkg[oldnan] = np.nan

    return bkg


