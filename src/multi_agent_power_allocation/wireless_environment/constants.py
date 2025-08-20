import numpy as np


# Number of APs
NUM_OF_AP = 1 
# Number of Devices K
NUM_OF_DEVICE = 10
# Number of Sub-6Ghz channels N and mmWave beam M
NUM_OF_SUB_CHANNEL = 4
if(NUM_OF_DEVICE == 10):
    NUM_OF_SUB_CHANNEL = 16
NUM_OF_BEAM = 4 
if(NUM_OF_DEVICE == 10):
    NUM_OF_BEAM = 16
# Noise Power sigma^2 ~ -169dBm/Hz
SIGMA_SQR = pow(10, -169/10)*1e-3
# Bandwidth Sub6-GHz = 100MHz, W_mW = 1GHz
# Bandwidth per subchannel W_sub = 100MHz/number of sub channel
W_SUB = 1e8/NUM_OF_SUB_CHANNEL
W_MW = 1e9
# Number of levels of quantitized Transmit Power
A = NUM_OF_SUB_CHANNEL
# Emitting power constraints 
P_SUM = pow(10,5/10)*1e-3*NUM_OF_DEVICE*2
# Frame Duration T_s 
T = 1e-3
# Packet size D = 8000 bit
D = 8000
# Number of frame
NUM_OF_FRAME = 10000
# LoS Path loss - mmWave
LOS_PATH_LOSS = np.random.normal(0,5.8,NUM_OF_FRAME+1)
# NLoS Path loss - mmWave
NLOS_PATH_LOSS = np.random.normal(0,8.7,NUM_OF_FRAME+1) 

# initialize position of AP.
# the AP was located at the central of the area
# the position of each AP is the constant
AP_POSITION = (0, 0)