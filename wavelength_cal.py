import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.visualization import ImageNormalize, ZScaleInterval
from matplotlib.backend_bases import FigureCanvasBase
import scipy.ndimage as nd

from os.path import exists

import helper_funcs as hf


class SnappingCursor:
    """
    A cross hair cursor that snaps to the data point of a line, which is
    closest to the *x* position of the cursor.

    For simplicity, this assumes that *x* values of the data are sorted.
    """

    def __init__(self, ax, line, curs=None):
        print('\n')
        self.ax = ax
        self.marker = []
        self.cal_marker, = ax.plot([], [], 'yv', ms=5, zorder=0)
        self.x, self.y = line.get_data()
        self.i = 0
        self._last_index = None
        self.locations = []
        self.curs = curs

    def on_mouse_move(self, event):
        if not event.inaxes:
            self._last_index = None
        elif (event.inaxes == self.ax) & (event.dblclick):
            x, y = event.xdata, event.ydata
            dd = (self.x - x)**2 + (self.y - y)**2
            index = np.argmin(dd)
            #index = min(np.searchsorted(self.x, x), len(self.x) - 1)
            if index == self._last_index:
                return  # still on the same data point. Nothing to do.
            self._last_index = index

            x = self.x[index]
            y = self.y[index]
            self.locations.append([x, y])
            self.ax.plot(x, y, 'r.')
            self.marker.append([x, y])
            self.ax.figure.canvas.draw()
            self.i += 1
            #print(f'{self.i} {x} {y}')

        if self.curs is not None:
            #             print('hey',len(self.locations),len(self.curs.locations))
            # rescale axes
            gtr2 = (len(self.locations) > 2) & (len(self.curs.locations) > 2)
            eq = len(self.locations) == len(self.curs.locations)
            if gtr2 & eq:
                pixel = np.array(self.locations)[:, 0]
                wave = np.array(self.curs.locations)[:, 0]
                wavesol = np.polyfit(np.sort(wave), np.sort(pixel)[::-1], 2)

                marker_w = self.curs.x*1
                marker_x = np.polyval(wavesol, marker_w)  # go from wave to pix
                marker_y = np.interp(marker_x, self.x, self.y)
                h = 1.05 * marker_y
                # print(wavesol)
                self.cal_marker.set_data(marker_x, h)
                self.ax.figure.canvas.draw()



def interactively_select_lines(cal, sl1=slice(None, None), order=2, threshold=0.05, size=5, backg_size=20,cal_dir=-1):
    """
    interactively select lines to calibrate with
    uses Hg/Ar list in current directory

    display the spectrum and the linelist

    After 3 peak/line pairs are selected it will display the current best fit

    cal: numpy array. if 2D it will take the mean(axis=0) to derive spectrom
    sl1: slice object indicating which rows to take


    """
    print('==  Find peaks  ==')
    if cal.ndim == 2:
        cal = np.nanmean(cal[sl1, :], axis=0)
    back = hf.findback1d(cal, s=backg_size)
    cal = cal - back
    peaks, cal_peak = hf.find_peaks(cal, threshold=threshold, size=size, centroid=True)

    print('== Interactive line selection ==')
    make_selection = True
    # this section blocks the terminal so user can interact with plot
    while make_selection:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9, 4))
        ax.set_title('Calibration spectrum')
        x = np.arange(len(cal))
        ax.plot(x, cal, lw=1)
        # cal_peak = np.interp(peaks,x,cal)
        line1, = ax.plot(peaks, cal_peak, 'ko', mec='w')
        ax.invert_xaxis()
        # fig.show()
        fig.suptitle('Double click lines. Press [ENTER] when done')

        # Plot and show the line lists
        hg = np.loadtxt('hgar_blue.txt')
        ar = np.loadtxt('argon_red.txt')

        I, w = np.loadtxt('nist_hg_I', usecols=(0, 1)).T
        Ihg = I[np.argmin(np.abs(hg - w[:, np.newaxis]), axis=0)]
        whg = w[np.argmin(np.abs(hg - w[:, np.newaxis]), axis=0)]

        I2, w2 = np.loadtxt('nist_ar_I', usecols=(0, 1)).T
        Iar = I2[np.argmin(np.abs(ar - w2[:, np.newaxis]), axis=0)]
        war = w2[np.argmin(np.abs(ar - w2[:, np.newaxis]), axis=0)]

        # fig2, ax2 = plt.subplots()
        ax2.set_title('Hg/Ar spectrum')

        colors = np.append(['b']*len(hg), ['r']*len(ar))
        line2, = ax2.plot(np.append(hg, ar), np.append(Ihg, Iar), '.')

        for i in range(len(w)):
            wi, Ii = w[i], I[i]
            ax2.plot([wi, wi], [0, Ii], 'k', lw=.5)

        ax2.plot(hg, Ihg, 'ko', mec='b')

        for i in range(len(w2)):
            wi, Ii = w2[i], I2[i]
            ax2.plot([wi, wi], [0, Ii], 'k', lw=.5)

        ax2.plot(ar, Iar, 'ko', mec='r')

        ax2.set_xlim(3500, 8000)

        line_cursor = SnappingCursor(ax2, line2)
        cid2 = fig.canvas.mpl_connect('button_press_event', line_cursor.on_mouse_move)
        cal_cursor = SnappingCursor(ax, line1, curs=line_cursor)
        cid1 = fig.canvas.mpl_connect('button_press_event', cal_cursor.on_mouse_move)

        def on_key(event):
            if event.key == 'enter':
                # unblock the terminal
                fig.canvas.stop_event_loop()
        cid3 = fig.canvas.mpl_connect('key_press_event', on_key)

        plt.show(block=False)

        print('Please select cooresponding lines to calibrate')
        # block the terminal
        fig.canvas.start_event_loop(timeout=-1)
        # now that we are back in the terminal,
        fig.canvas.mpl_disconnect(cid1)
        fig.canvas.mpl_disconnect(cid2)
        fig.canvas.mpl_disconnect(cid3)

        key = input("And then press any key to continue [q hard quits w/o saving]:")

        if key == 'q':
            plt.close(fig)
            return 0, 0, ax, fig

        if len(cal_cursor.locations) == len(line_cursor.locations):
            make_selection = False
            # we will continue the loop
        else:
            make_selection = True

    # left the blocking section
    # breakpoint()
    cal_locs = np.array(cal_cursor.locations)[:, 0]
    line_locs = np.array(line_cursor.locations)[:, 0]

    print(f'Deriving wavelength solution. Order={order}')
    wavesol = hf.wavelength_cal(np.sort(cal_locs)[::cal_dir], np.sort(line_locs), order=order)
    x = np.arange(len(cal))
    wave = np.polyval(wavesol, x)

    # all the rest of this is plotting
    ax.clear()
    # plot the calibrated spectrum
    ax.plot(wave, cal, lw=1)

    # show the wavelength solution. should look mostly straight
    ax.plot(wave, hf.scale_ptp(wave) * cal.max()/2, 'r', lw=1)

    linelist = np.append(hg, ar)

    # Plot and show the line lists
    hg = np.loadtxt('hgar_blue.txt')
    ar = np.loadtxt('argon_red.txt')

    # get the line intensities from NIST
    I, w = np.loadtxt('nist_hg_I', usecols=(0, 1)).T
    Ihg = I[np.argmin(np.abs(hg - w[:, np.newaxis]), axis=0)]
    whg = w[np.argmin(np.abs(hg - w[:, np.newaxis]), axis=0)]

    I2, w2 = np.loadtxt('nist_ar_I', usecols=(0, 1)).T
    Iar = I2[np.argmin(np.abs(ar - w2[:, np.newaxis]), axis=0)]
    war = w2[np.argmin(np.abs(ar - w2[:, np.newaxis]), axis=0)]

    for i in range(len(whg)):
        wi, Ii = whg[i], Ihg[i]/Ihg.max()
        ax.plot([wi, wi], [0, cal.max()*Ii], 'k', lw=.5)

    for i in range(len(war)):
        wi, Ii = war[i], Iar[i]/Iar.max()
        ax.plot([wi, wi], [0, cal.max()*Ii], 'k', lw=.5)

    ax.set_xlim(wave.min()*0.98, wave.max()*1.02)

    # draw the new canvas
    fig.canvas.draw()

    return cal_locs, line_locs, cal, ax, fig


def wavelength_cal(cal_file, recal='n', threshold=0.05, size=5, order=2, rect_order=2, rect_slice=None, cal_dir=-1):
    """
    rect_slice = bottom, top, left, right # bottom and left are the smaller numbers
    cal_dir = -1 # 1 if wavelength increases with pixel

    if cal_file is a file, check if it has already been reduced
    if not run the calibration.
    if cal_file is an array, then run the calibration, but no output will be saved.
    """

    if isinstance(cal_file, str):
        fname = f"{cal_file.replace('.FIT','.tsv')}"
        calibration_occured = False
        run_calibration = (exists(fname) & (recal == 'y')) | (not exists(fname))
    else:
        run_calibration = True
    if run_calibration:
        print('===== Begin Calibration =====')
        cal = fits.getdata(cal_file).astype(float)
        cal = cal - nd.median(cal)

        print('==  Rectify 2D Spectrum  ==')
        if rect_slice is not None:
            sl1 = slice(rect_slice[0], rect_slice[1])
            sl2 = slice(rect_slice[2], rect_slice[3])
            rect_sol, full_frame_solution = hf.rectify_ccd(
                cal - nd.gaussian_filter1d(cal, 20, axis=1), order=rect_order)
        else:
            sl1 = slice(None, None)
            sl2 = slice(None, None)
            rect_sol, full_frame_solution = hf.rectify_ccd(cal, order=rect_order)
        cal = hf.shift_row_interp(cal, rect_sol)

        # Go through the process of selecting lines ineractively
        cal_locs, line_locs, cal, ax, fig = interactively_select_lines(
            cal, sl1, order=order, threshold=threshold, size=size,cal_dir=cal_dir)

        if not hasattr(cal_locs,'__iter__'):
            print('\n========= Something went wrong ========= ')
            print('Returning without saving anything')
            return 0, 0, 0, 0

        #print(f'Deriving wavelength solution. Order={order}')
        # breakpoint()
        wavesol = hf.wavelength_cal(np.sort(cal_locs)[::cal_dir], np.sort(line_locs), order=order)
        x = np.arange(len(cal))
        wave = np.polyval(wavesol, x)

        # calibration is done
        calibration_occurred = True

        # check if the user wants to recalibrate
        recal = input(f'Would you like to recalibrate {fname}? [y,n]: ')
        if recal == 'n':
            plt.close(fig)

    # if a filename was given, and found, then
    elif exists(fname):
        # File exists so get the information
        wave, cal = np.loadtxt(fname, skiprows=3).T
        with open(fname, 'r') as f:
            p = f.readline().split()[1:]
            p = [float(i) for i in p]
            w = f.readline().split()[1:]
            w = [float(i) for i in w]
        return p, wave, cal, w

    if recal.lower() == 'y':
        plt.close()
        wavelength_cal(cal_file, recal=recal)
    elif calibration_occurred:
        plt.close()
        out = list(zip(wave, cal))
        header = " ".join([f"{w} " for w in wavesol])
        header = header + "\n" + " ".join([f"{w} " for w in rect_sol])
        header = header + "\nwave cal"
        breakpoint()
        np.savetxt(fname, out, fmt=('%-9.3f\t%-10.3f'), header=header)
        return wavesol, wave, cal, rect_sol
    else:
        return 0, 0, 0, 0


class HorizontalCursor:
    """
    a horizontal range
    """

    def __init__(self, ax):
        self.ax = ax
        bottom, top = self.ax.get_ylim()
        self.top = top
        self.bottom = bottom
        self.top_line = ax.axhline(color='r', ls='--', lw=1.5)
        self.bot_line = ax.axhline(color='b', ls='--', lw=1.5)
        self.clicknum = 1

    def on_mouse_click(self, event):
        if not event.inaxes:
            pass
        elif event.inaxes == self.ax:
            # print(event)
            y = event.ydata
            if self.clicknum < 0:
                self.bottom = y
                self.bot_line.set_ydata(y)
            else:
                self.top = y
                self.top_line.set_ydata(y)
            self.ax.figure.canvas.draw()
            self.clicknum = -self.clicknum
    def sort(self):
        t = max(self.top, self.bottom)
        b = min(self.top, self.bottom)
        self.top = t
        self.bottom = b
        pass


def image_range_picker(array):

    fig, ax = plt.subplots(1, 1)

    arr = hf.scale_ptp(array)
    vmin, vth, vmax = np.percentile(arr, [1, 50, 99])
    # norm = mc.SymLogNorm(vth, vmin=vmin, vmax=vmax)
    ax.imshow(arr, vmin=vmin,vmax=vmax,cmap='viridis')
    line_cursor = HorizontalCursor(ax)

    cid = fig.canvas.mpl_connect('button_press_event', line_cursor.on_mouse_click)

    def on_key(event):
        if event.key == 'enter':
            line_cursor.sort()
            fig.canvas.stop_event_loop()
            plt.close()
    cid3 = fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show(block=False)

    fig.canvas.start_event_loop(timeout=-1)
    print(int(line_cursor.top),int(line_cursor.bottom))
# Plot and show Calibration spectrum
# cal_file = '2021_10_06_test_data/cal_lamp1.FIT'
# os.system('rm -f 2021_10_06_test_data/cal_lamp1.tsv')
# wavesol, wave, cal, rect_sol = wavelength_cal(cal_file,size=5)
