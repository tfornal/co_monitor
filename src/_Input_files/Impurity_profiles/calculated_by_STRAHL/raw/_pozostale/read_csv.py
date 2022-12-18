import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt

def smooth(x, window_len=11, window='hanning'):
    if x.ndim != 1:
        print("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        print("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        print('this window is not defined')
    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')
    y = np.convolve(w / w.sum(), s, mode='valid')
    y = y[int(0.5 * window_len):int(len(y) - 0.5 * window_len) + 1]
   
    return y


def read_csv(conv_or_diff):

    interpolation_step = 141    

    if conv_or_diff in ["D", "d"]:
        file_name = "profilD.csv"
    elif conv_or_diff in ["V", "v"]:
        file_name = "profilV.csv"
        
    file = pd.read_csv(file_name)
    reff = file.iloc[:, 0]
    profil = file.iloc[:, 1]
    
    interp_func = interp1d(reff, profil, fill_value = "extrapolate")
    interpolated_profile = interp_func(np.linspace(reff[0], reff.iloc[-1], num = interpolation_step, endpoint=True))
    
    interpolated_reff  = np.linspace(reff[0], reff.iloc[-1], num = interpolation_step, endpoint=True)
    
    if conv_or_diff == "D":
        interpolated_profile = interpolated_profile + 2*abs(interpolated_profile[0])

    

    interpolated_profile = smooth(interpolated_profile)

    def plot():
        plt.plot(interpolated_reff, interpolated_profile)
        plt.show()
    plot()
    
    np.savetxt(f"{conv_or_diff.capitalize()}.txt", np.c_[interpolated_reff, interpolated_profile])
    
    return interpolated_profile
    
    

def main():
    read_csv("D")

if __name__ == "__main__":
    main()
    
    
    
    