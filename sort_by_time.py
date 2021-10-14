import os, glob, sys

import datetime as dt

#data_dir = '2021_10_06_test_data'

def getheader_val(fname,card_name):
    """
    Pure python function to get header values
    from a fits file
    
    """
    with open(fname,'rb') as f:
        line = f.read(80)
        while line.strip()!=b'END':
            if line[:7]!=b'HISTORY':
                card_value = line.split(b'/')[0]
                card, value = card_value.split(b'=')
                if card.strip().decode()==card_name:
                    return value.strip().decode().replace("'","")
            line = f.read(80)
            

if __name__ == "__main__":
    print('\nListing calibration and date files\n')
    print('Paste a line into the terminal and run')
    print('> python plot_and_reduce.py -c $cal_file -d $data_file\n')
    data_dir = sys.argv[1]
    data_dir = os.path.realpath(data_dir)
        
    files = glob.glob(os.path.join(data_dir,'*.FIT'))

    date = []
    for f in files:
        date_obs = getheader_val(f,'DATE-OBS')
        date.append(dt.datetime.fromisoformat(date_obs,))
        
    argsort = lambda s: sorted(range(len(s)), key=lambda k: s[k])

    srt = argsort(date)

    with open('reduction_script','w') as fred:
        fred.write('#Make sure to check no images were left. Removed files with image in the name')
        fred.write('#You may need to add -t and -b parameters for some targets. so look at the output plots')
        fred.write('#Changes can be made directly in the reduction_script')
        for s in srt:
            f = os.path.relpath(files[s])
            obj = getheader_val(f,'OBJECT').strip().lower()
            npix = getheader_val(f,'NAXIS2').strip().lower()
            if obj == 'cal':
                cal = f
            elif (npix=='510'):
                print(f'cal_file={cal};data_file={f}')
                line = f'python plot_and_reduce.py -c {cal} -d {f} --batch\n\n'
                fred.write(line)