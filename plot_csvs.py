import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
from os.path import basename


# Plot data from first two columns of csv files in directory
def plot_csvs(directory, file_names, x_label='wavelength', y_label='counts', title=None):
    for file_name in file_names:
        data = np.loadtxt(directory + file_name, delimiter=',')
        plt.figure()
        plt.plot(data[:, 0], data[:, 1])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if title is None:
            title = file_name.split('.')[0]
        plt.title(title)
        plt.savefig(directory + file_name + '_plot.png')


if __name__ == '__main__':
    # use parser to get directory and file names
    parser = argparse.ArgumentParser(description='Plot csv files in directory')
    parser.add_argument('directory', default='.',help='directory containing csv files')
    parser.add_argument('file_names', default=None,type=str,help='file names of csv files to plot')

    args = parser.parse_args()
    if args.directory[-1] != '/':
        args.directory += '/'
    if args.file_names == "all":
        file_names = [basename(i) for i in glob.glob(args.directory + '*.csv')]
    else:
        file_names = args.file_names.split(',')

    plot_csvs(args.directory, file_names)
