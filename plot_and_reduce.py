import argparse
import warnings
import os
import numpy as np
import astropy.stats as astats
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import colors as mc


import helper_funcs as hf

hg = np.loadtxt('hgar_blue.txt') / 10
ar = np.loadtxt('argon_red.txt') / 10


def plot_reduce(cal_file, data_file, cal_threshold=0.05, bottom=None, top=None, rectify=True, order=2,
                plot=True, recal='n', cosmic_rays=False, save=False, slice_fwhm=1.5, size=5,sbig=False,diffuse=False):
    print('===READ IN DATA===')
    cal = fits.getdata(cal_file).astype(float)
    cal = cal + float(hf.getheader_val(cal_file, 'PEDESTAL'))

    data = fits.getdata(data_file).astype(float)
    obj = hf.getheader_val(data_file, 'OBJECT')
    data = data + float(hf.getheader_val(data_file, 'PEDESTAL'))

    print('Get wavelength solution')
    out = hf.get_wavelength_cal(cal_file, order=order,
                                threshold=cal_threshold,
                                size=size, recal=recal)
    p, wavesol, cal_spec, rect_sol = out

    with plt.rc_context(rc={'image.origin': 'upper'}):
        if plot:
            mosaic = """AABB;CCDD;.II.;EEFF;GGHH"""
            fig, axs = plt.subplot_mosaic(mosaic, gridspec_kw={'height_ratios': [1, 1, .7, 1, 1.25]}, figsize=(12, 16))
        else:
            mosaic = """GH"""
            fig, axs = plt.subplot_mosaic(mosaic, figsize=(10, 4))
            axs['A'] = None
            axs['B'] = None
            axs['C'] = None
            axs['D'] = None
            axs['E'] = None
            axs['F'] = None
            axs['I'] = None

        # rect_sol, full_frame_solution = rectify_ccd(cal,plot=plot,ax=axs['A']);
        print('Rectify')
        if rectify:
            cal = hf.shift_row_interp(cal, rect_sol, plot=plot, axs=(axs['A'], axs['B']))
            data = hf.shift_row_interp(data, rect_sol, plot=plot, axs=(axs['C'], axs['D']))
        else:
            hf.shift_row_interp(cal, rect_sol, plot=plot, axs=(axs['A'], None))
            hf.shift_row_interp(data, rect_sol, plot=plot, axs=(axs['C'], None))

        print('Extract spectra')
        if diffuse:
            sl = slice(1,data.shape[0]-1)  # full range
            plt.delaxes(axes['I'])
        else:
            sl = hf.specextract(data, bottom=bottom, top=top, plot=plot, ax=axs['I'], slice_fwhm=slice_fwhm,sbig=sbig)

        # plot the section we will extract
        if plot:
            # vmin, vmax = np.percentile(cal, [50, 99])
            # norm = mc.SymLogNorm(100, vmin=vmin, vmax=vmax)
            norm = hf.make_norm(cal)
            ax = axs['E']
            ax.imshow(cal, norm=norm, cmap='viridis')
            ax.axhline(sl.start, color='r')
            ax.axhline(sl.stop, color='r')
            ax.set_title('Calibration')
            ax = axs['F']
            # vmin, vmax = np.percentile(data, [50, 99])
            # norm = mc.SymLogNorm(100, vmin=vmin, vmax=vmax)
            norm = hf.make_norm(data)
            ax.imshow(data, norm=norm, cmap='viridis')
            ax.axhline(sl.start, color='r')
            ax.axhline(sl.stop, color='r')
            ax.set_title('Target')

        # Calibration spectrum and peaks for plotting
        cal_spec = np.nanmean(cal, axis=0)
        back = hf.findback1d(cal_spec, s=20)
        peaks, peaks_y = hf.find_peaks(cal_spec - back, threshold=cal_threshold, size=size,)  # put in wavelength order
        linelist = np.append(hg, ar)
        new = np.polyval(p, peaks)
        matched = linelist[np.argmin(np.abs(new - linelist[:, np.newaxis]), axis=0)]

        ax = axs['G']
        # Plot wavelength cal
        ax.plot(wavesol, cal_spec, lw=1)
        x = np.arange(len(cal_spec))
        # ax.plot(np.polyval(p,peaks),peaks_y,'r.')
        ax.plot(wavesol, back, lw=1)
        for i in np.polyval(p, peaks):
            ax.axvline(i, color='0.5', zorder=0, lw=.5)
        ax.invert_xaxis()
        ax.set_title('wavelength cal')
        ax.set_xlabel('Wavelength [Å]')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if cosmic_rays:
                cr = hf.get_cosmic_rays(data)
                im2 = np.zeros(cr.shape + (4,))
                im2[:, :, 0] = 1
                im2[:, :, 3] = cr * 1
                if axs['F'] is not None:
                    axs['F'].imshow(im2)
                spec = np.ma.mean(np.ma.masked_array(data, cr)[sl, :], axis=0)
            else:
                spec = np.nanmean(data[sl, :], axis=0)

        # noise = np.nanstd(data[sl,:],axis=0)/np.sqrt(top-bot)

            N = np.sum(np.isfinite(data[sl,:]),axis=0)
            noise = astats.sigma_clipped_stats(data, sigma=3,)[-1] / np.sqrt(N)
            noise = np.sqrt(spec + noise**2)
        # breakpoint()
        ax = axs['H']
        hf.errorbar_fill(wavesol, spec, noise, mid=True, alpha=1, lw=.5, ax=ax)
        # plt.plot(wavesol,data[sl,:].T,'k',lw=.5)

        ax.set_xlabel('Wavelength [Å]')
        ax.set_ylabel('Uncalibrated data units')
        ax.set_title(f'Spectrum of {obj}')

        out = np.array(list(zip(wavesol, spec, noise, cal_spec-back)))

        if save:
            fname = f"{data_file.replace('.FIT','.csv')}"
#             with open(fname,'w') as f:
#                 f.write('wavelength\tspectrum\terror\tcal\n')
#                 for i in out:
#                     f.write('{:<9.3f}\t{:>10.3f}\t{:>10.3f}\t{:>10.3f}\n'.format(*i))
            np.savetxt(fname, out, fmt=('%-9.3f , %-10.3f , %-10.3f , %-10.3f'), header='wave spec err cal')

        return (wavesol, spec, noise), fig


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Reduce a spectrum from the spectrograph on the Clay Telescope (Harvard University)")

    parser.add_argument("-c", "--cal", help="""The calibration file""")

    parser.add_argument("-d", "--data", help="The spectrum we want reduced")

    parser.add_argument("-s", "--no-sbig", action='store_false', help="Don't reduce data using 20 pixel cut (like required for SBIG Spectra software")

    parser.add_argument("--diffuse", action='store_true', help="Use entire slit (for diffuse object spectra)")

    parser.add_argument("-b", "--bottom", default=None, type=int, help="lower index for spectrum boundary")

    parser.add_argument("-t", "--top", default=None, type=int, help="higher index for spectrum boundary")

    parser.add_argument("--threshold", default=0.05, type=float,
                        help="Threshold level used for defining getting peaks for calibration")

    parser.add_argument("--width", default=1.5, type=float, help="Number of FWHMs to use for extracting spectrum")

    parser.add_argument("--plot", action='store_true', help="Plot only the spectrum")

    parser.add_argument("-r", "--recalibrate", action='store_true',
                        help="Force recalibration. Otherwise it will silently use a previous calibration if one exists")


    parser.add_argument("--batch", action='store_true', help="Don't open images")

    parser.add_argument("-o", "--order", default=2, type=int, help="Which order should wavelength calibration use")

    parser.add_argument("--zap-cosmic-rays", default=False, help=""""Get rid of cosmic rays.
                                                        This uses simple algorithm that could clip bright lines.
                                                        Best used only for long integrations of diffuse nebular emission""")

    args = parser.parse_args()

    cal_file = args.cal
    data_file = args.data

    threshold = args.threshold

    bottom = args.bottom
    top = args.top

    rectify = True  # args.dont_rectify
    save = True  # args.dont_save

    zap_cosmic_rays = args.zap_cosmic_rays

    plot = not args.plot

    order = args.order

    hg = np.loadtxt('hgar_blue.txt') / 10
    ar = np.loadtxt('argon_red.txt') / 10

    sbig = ~args.no_sbig

    if not ((bottom is None) or (top is None)):
        sbig = False

    diffuse = args.diffuse

    slice_fwhm = args.width

    if args.recalibrate:
        recal = 'y'
    else:
        recal = 'n'

    out, fig = plot_reduce(cal_file, data_file, cal_threshold=threshold, bottom=bottom, top=top, order=order,
                           rectify=rectify, cosmic_rays=zap_cosmic_rays, plot=plot, save=save, slice_fwhm=slice_fwhm, recal=recal,sbig=sbig,diffuse=diffuse)

    fname = f"{data_file.replace('.FIT','_big.png')}"
    fig.savefig(fname)

    if not args.batch:
        os.system(f'open {fname}')
