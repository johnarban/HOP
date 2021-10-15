import argparse
import warnings
import numpy as np
from scipy import ndimage
import astropy.stats as astats
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.modeling.fitting import SimplexLSQFitter
from astropy.modeling.models import Gaussian1D, Chebyshev1D

import helper_funcs as hf
import wavelength_cal as wc


def reduce(cal_file, data_file, cal_threshold=0.05, bottom = None, top = None, rectify=True,
           plot=False,clip_cal=False,cosmic_rays=False,save=True,order=2,size=5,slice_fwhm=1.5,
           recal='n'):
    """
    reduce(cal_file, data_file, bottom = #, top = #, save = True, order = 2)

    """
    cal = fits.getdata(cal_file).astype(float)
    cal = cal + float(hf.getheader_val(cal_file, 'PEDESTAL'))


    data = fits.getdata(data_file).astype(float)
    obj = hf.getheader_val(data_file,'OBJECT')
    data = data + float(hf.getheader_val(data_file, 'PEDESTAL'))

    # 1) Rectify data & Get wavelength calibration
    out = hf.get_wavelength_cal(cal_file, order = order,
                                threshold=cal_threshold,
                                size=size,recal=recal)
    p, wavesol, cal_spec, rect_sol = out

    # y, x_line  = np.indices(cal.shape)
    # full_frame_solution = x_line - np.polyval(p_disp,y)
    # rect_sol, full_frame_solution = rectify_ccd(cal);
    if rectify:
        cal = hf.shift_row_interp(cal,rect_sol)
        data = hf.shift_row_interp(data,rect_sol)

    # remove offset/bias


    # 2) Extract selection for spectral regions
    sl = hf.specextract(data,bottom=bottom,top=top,slice_fwhm=slice_fwhm)

    # 3) Get wavelength calibration
    # p, wavesol, cal_spec = get_wavelength_cal(cal[sl,:],order=order)


    # 4) Some targets are cosmic rays (this is very simplistic)
    if cosmic_rays:
        cr = hf.get_cosmic_rays(data)
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

    parser.add_argument("-r","--recalibrate",action='store_true',help="Force recalibration. Otherwise it will silently use a previous calibration if one exists")

    parser.add_argument("--width",default = 1.5,type=float,help="Number of FWHMs to use for extracting spectrum")

    # parser.add_argument("-s","--dont-save",action='store_true',help="Don't save the spectra?")

    parser.add_argument("--batch",action='store_true',help="Don't open images")

    parser.add_argument("-o","--order",default=2,type=int,help="Don't open images")

    parser.add_argument("--zap-cosmic-rays",default=False,help=""""Get rid of cosmic rays.
                                                        This uses simple algorithm that could clip bright lines.
                                                        Best used only for long integrations of diffuse nebular emission""")

    args = parser.parse_args()

    cal_file = args.cal
    data_file = args.data

    threshold = args.threshold

    bottom = args.bottom
    top = args.top

    rectify = True #args.rectify

    zap_cosmic_rays = args.zap_cosmic_rays

    plot = args.plot

    order = args.order

    save = True #not args.dont_save

    slice_fwhm = args.width

    hg = np.loadtxt('hgar_blue.txt') / 10
    ar = np.loadtxt('argon_red.txt') / 10

    if args.recalibrate:
        recal = 'y'
    else:
        recal = 'n'

    out = reduce(cal_file, data_file, cal_threshold=threshold, bottom = bottom, top = top, order = order, rectify=rectify, cosmic_rays=zap_cosmic_rays, save=save, slice_fwhm=slice_fwhm, recal=recal)


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

