"""
Created on Wed Sep  2 23:12:19 2020

@author: filipe

Digital Signal Processing
Computation of the discrete fourier transform using three different methods.
Auxiliary functions implemented: 
- zero pad data when data length is not power of 2
- bit reverse the index of an array (length power of 2)
- bit reverse number

Console output:
- information on data length and general errors
- time to execute each method of computation

version 5 - github final
"""

from pandas import read_csv
from time import time
import numpy as np

#%% Auxiliary functions
def zero_pad(data):
    # Check if data is a power of 2 and adds zeros (zero padding) if necessary
    N_data = len(data)
    logN = np.log2([N_data])[0]
    
    if (N_data & (N_data - 1) == 0) and N_data != 0:
        # Data length already power of 2
        print('INFO: Data length is power of 2')
        return data
    else:
        # Finds the next power of 2 to fill the data array with zeros
        next_power_of_2 = int(np.ceil(logN))
        target_data_len = 2 ** next_power_of_2
        missing_points_N = target_data_len - N_data
        new_data = np.append(data, np.array([0]*missing_points_N)) # Appends zeros to the end of data
        print('INFO: Data length is NOT power of 2. New data length:', target_data_len)
        
        return new_data

def bit_reverse_array(data_len):
    # Receives the array length. MUST BE POWER OF 2!!
    # Returns array with the bit reversed index
    logN = int(np.log2([data_len])[0])
    binary_len = '0' + str(logN) +'b'
    output = np.array([], dtype=np.int32) # Output array (int)
    
    for i in range(data_len):
        i_bin = format(i, binary_len)   # i to binary
        i_reverse = i_bin[::-1]         # reverse bits
        val_reverse = int(i_reverse, 2) # binary to int base 2
        output = np.append(output, val_reverse)

    return output

#%% DFT computation main functions
def dft(data):
    # Implementation of the Discrete Fourier Transform through direct
    # use of the expression X[k] = SUM{n=0 -> N-1} x[n]Wn^(nk)

    # Data consistency check
    if len(data) == 0:
        print('Data length = 0')
        return np.array([])
    
    N_data = len(data)
    Wn = np.exp(-1j*2*np.pi/N_data)
    Xk = np.array([], dtype = 'complex')
    
    start_time = time()
    
    for k in range(N_data):
        sum_xk = 0    
        for n in range(N_data):
            sum_xk += data[n] * Wn**(k*n)

        Xk = np.append(Xk, sum_xk)

    execution_time = time() - start_time
    print('INFO: DFT Direct Computation execution time (s):', np.around(execution_time, 9))
    
    return Xk

#%% Implementation of the FFT using time decimation
def fft_td(data):
    
    # Data consistency check
    if len(data) == 0:
        print('Data length = 0')
        return np.array([])

    # Constants
    N = len(data)    
    Wn2 = np.exp(-1j*np.pi*2/(N/2))
    Wn  = np.exp(-1j*np.pi*2/N)
    # Data output
    Gk = np.array([], dtype = 'complex')
    Hk = np.array([], dtype = 'complex')
    
    start_time = time()
    
    for k in range(N):
        ii, jj = 0, 0
        for n in range(N//2):
            ii = ii + data[2*n] * Wn2 ** (k*n)   # Even values
            jj = jj + data[2*n+1] * Wn2 ** (k*n) # Odd values

        jj = jj * Wn**k    
        Gk = np.append(Gk, ii)
        Hk = np.append(Hk, jj)

    Xk = Gk + Hk
    execution_time = time() - start_time
    print('INFO: FFT Time Decimation execution time (s):', np.around(execution_time, 9))

    return Xk

#%% Implementation of the Cooley-Tukey algorithm
def fft_bf(data):
    
    data = data.astype(complex)
    verbose = False
    N = len(data)
    bits = int(np.log2([N])[0])
    Wn = np.exp(-2j*np.pi/N)
    
    reversed_index = bit_reverse_array(N) # bit reversed index
    X_m = np.array([], dtype = 'complex')
    X_m = data[reversed_index]            # Stage m-1 assumes data with reversed index
    
    start_time = time()
    
    for m in range(int(bits)):
        if verbose: print('Estágio:', 1)
        # Values for each stage
        blocks = N // (2**(m+1))       # Number of blocks   
        bf_per_block = 2**m            # Butterflies per block
        bf_size = 2**m                 # Butterfly size
        block_step = 2**(m+1)          # Length of the block in lines
        twiddle_step = N // (2**(m+1)) # Twiddle power increment

        for block in range(blocks):
            # Loop through blocks
            if verbose: print('   Bloco:', block)
            
            block_start = block*block_step
            for bf in range(bf_per_block):
                # Loop through butterflies
                if verbose: print('     Borboleta:', bf)
                p = block_start + bf
                q = p + bf_size
                r = bf * twiddle_step
                
                P_temp = X_m[p]
                Q_temp = X_m[q] * Wn**r
                
                X_m[p] = P_temp + Q_temp
                X_m[q] = P_temp - Q_temp
                
                if verbose: print('        p: {}  q: {}'.format(p,q))
                if verbose: print('        twd:', r)

    execution_time = time() - start_time
    print('INFO: FFT DIT Butterfly Decimation execution time (s):', np.around(execution_time, 9))
                
    return X_m
