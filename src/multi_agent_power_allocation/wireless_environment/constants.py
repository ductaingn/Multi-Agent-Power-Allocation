import numpy as np


# Number of APs
NUM_OF_AP = 1
# Number of Devices K
NUM_OF_DEVICE = 10
# Number of Sub-6Ghz channels N and mmWave beam M
NUM_OF_SUB_CHANNEL = 4
if NUM_OF_DEVICE == 10:
    NUM_OF_SUB_CHANNEL = 16
NUM_OF_BEAM = 4
if NUM_OF_DEVICE == 10:
    NUM_OF_BEAM = 16
# Noise Power sigma^2 ~ -169dBm/Hz
SIGMA_SQR = pow(10, -169 / 10) * 1e-3
# Bandwidth Sub6-GHz = 100MHz, W_mW = 1GHz
# Bandwidth per subchannel W_sub = 100MHz/number of sub channel
W_SUB = 1e8 / NUM_OF_SUB_CHANNEL
W_MW = 1e9
# Number of levels of quantitized Transmit Power
A = NUM_OF_SUB_CHANNEL
# Emitting power constraints
P_SUM = pow(10, 5 / 10) * 1e-3 * NUM_OF_DEVICE * 2
# Frame Duration T_s
T = 1e-3
# Packet size D = 8000 bit
D = 8000
# Number of frame
NUM_OF_FRAME = 10000

def _initialize_path_loss_constants(
        n_frame: int = NUM_OF_FRAME, 
        n_devices: int = NUM_OF_DEVICE,
        rng: np.random.Generator | None = None
    ) -> tuple:
    """
    Initialize path loss constants with optional seed control.
    
    Parameters
    ----------
    n_frame : int, optional
        Number of frames. Default is NUM_OF_FRAME.
    n_devices : int, optional
        Number of IoT devices. Default is NUM_OF_DEVICE.
    rng : np.random.Generator | None, optional
        numpy random generator
    Returns
    -------
    tuple
        (LOS_PATH_LOSS, NLOS_PATH_LOSS) arrays
    """
    if rng is None:
        los = np.random.normal(0, 5.8, size=(n_frame + 1, n_devices))
        nlos = np.random.normal(0, 8.7, size=(n_frame + 1, n_devices))
        return los, nlos
    
    los = rng.normal(0, 5.8, size=(n_frame + 1, n_devices))
    nlos = rng.normal(0, 8.7, size=(n_frame + 1, n_devices))

    return los, nlos

# Map specs
AP_RANGE = 142
MAP_SIZE = (400, 400)
