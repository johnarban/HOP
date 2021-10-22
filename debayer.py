from astropy.io import fits
import numpy as np
import sys
def debayer(f):
    bayer = fits.getdata(f).astype(float)
    #  RG  #
    #  GB  #
    r = bayer[::2,::2]
    g = (bayer[1::2,::2]+bayer[::2,1::2])/2
    b = bayer[1::2,1::2]

    rgb = np.zeros(r.shape+(3,))
    rgb[:,:,0] = r / 65535
    rgb[:,:,1] = g / 65535
    rgb[:,:,2] = b / 65535
    return rgb


if __name__ == "__main__":
    
    f = sys.argv[1]
    
    rgb = debayer(f)
    
    import matplotlib.pyplot as plt
    
    plt.imshow(rgb)
    plt.axis('off')
    plt.savefig(f.replace('.FIT','.png')
    