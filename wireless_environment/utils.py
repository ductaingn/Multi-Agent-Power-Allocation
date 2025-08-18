"""
Utility functions for wireless environment simulation.
This module provides functions to generate device positions, compute path loss, rates, e.t.c.
"""
import numpy as np
import pickle
from typing import Union


def generate_devices_positions(num_of_device:int) -> np.ndarray:
    """
    Generate positions for devices in the environment.

    Parameters
    ----------
    num_of_device : int
        Number of devices to generate positions.
    
    Returns
    -------
    np.ndarray
        Array of device positions, each position is a list of [x, y] coordinates.
    """
    device_positions = [
        [0, 20],     # Device 1
        [20, 0],     # Device 2 (Blocked by obstacle)
        [-85, -80],  # Device 3
        [-45, 40],   # Device 4
        [10, -70],   # Device 5
        [-40, -20],  # Device 6 (Blocked by obstacle)
        [-40, 15],   # Device 7
        [60, 55],    # Device 8
        [45, 5],     # Device 9
        [50, -40],   # Device 10
        [40, 60],    # Device 11 (Blocked by obstacle)
        [-20, -60],  # Device 12
        [-20, 80],   # Device 13
        [20, -40],   # Device 14 (Blocked by obstacle)
        [-80, 80]    # Device 15        
    ]
        
    return np.array(device_positions[:num_of_device], dtype=float)


def path_loss_sub(distance):
    """
    Path loss model for Sub-6GHz connections.

    Parameters
    -----------
    distance : float 
        Distance to the AP in meters.

    Returns
    -----------
    path_loss : float
        Path loss in dB.

    """
    return 38.5 + 30*(np.log10(distance))


def path_loss_mW_los(distance, x):
    """
    Path loss model for mmWave connections in Line-of-Sight conditions.

    Parameters
    -----------
    distance : float
        Distance to the AP in meters.
    x : float
        Random variable for path loss variation.

    Returns
    -----------
    path_loss : float
        Path loss in dB.
    """
    return 61.4 + 20*(np.log10(distance)) + x

def path_loss_mW_nlos(distance, x):
    """
    Path loss model for mmWave connections in Non-Line-of-Sight conditions.

    Parameters
    -----------
    distance : float
        Distance to the AP in meters.
    x : float
        Random variable for path loss variation.

    Returns
    -----------
    path_loss : float
        Path loss in dB.
    """
    return 72 + 29.2*(np.log10(distance)) + x


# Main Transmit Beam Gain G_b
def G(eta=5*np.pi/180, beta=0, epsilon=0.1):
    """
    Calculate the main transmit beam gain.
    
    Parameters
    ----------
    eta : float, optional
        Narrowest beamwidth in radians (default is 5 degrees converted to radians).
    beta : float, optional
        Line-of-Sight diretion from AP to device (default is 0).
    epsilon : float, optional
        Size lobe beam gain (default is 0.1). 
    
    Returns
    -------
    float
        Main transmit beam gain.
    """
    return (2*np.pi-(2*np.pi-eta)*epsilon)/(eta)

def generate_h_tilde_device_channel(amount:int, mu:float, sigma:float)->np.ndarray:
    """
    Complex channel coefficient of a pair of subchannel/beam and a device.
        h = h_tilde * 10^(-pathloss/20)
        h_tilde = (a + b*i)/sqrt(2)
    
        in which a and b is random value from a Normal(mu, sigma) distribution

    Parameters
    ----------
    amount : int
        Number of channel coefficients to generate.
    mu : float, optional
        Mean of the normal distribution (default is 0).
    sigma : float, optional
        Standard deviation of the normal distribution (default is 0.1).
    
    Returns
    -------
    h_tilde : np.ndarray
        Array of complex channel coefficients.
    """
    re = np.random.normal(mu, sigma, amount)
    im = np.random.normal(mu, sigma, amount)
    h_tilde = []
    for i in range(amount):
        h_tilde.append(complex(re[i], im[i])/np.sqrt(2))
    return np.array(h_tilde)


def generate_h_tilde(num_timestep:int, num_device:int, num_subchannel:int, num_beam:int, mu:float=0, sigma:float=1, save:bool=True, save_path:str='environment/scenario_1/h_tilde.pickle')->np.ndarray:
    """
    Generate complex channel coefficients for multiple devices, subchannels, and beams over a specified number of timesteps.

    Parameters
    ----------
    num_timestep : int
        Number of timesteps for which to generate channel coefficients.
    num_device : int
        Number of devices for which to generate channel coefficients.
    num_subchannel : int
        Number of subchannels for which to generate channel coefficients.
    num_beam : int
        Number of beams for which to generate channel coefficients.
    mu : float, optional
        Mean of the normal distribution for generating channel coefficients (default is 0).
    sigma : float, optional
        Standard deviation of the normal distribution for generating channel coefficients (default is 1).
    save : bool, optional
        Whether to save the generated channel coefficients to a file (default is True).
    save_path : str, optional
        Path to save the generated channel coefficients if `save` is True (default is 'environment/scenario_1/h_tilde.pickle'). 

    Returns
    -------
    h_tilde : np.ndarray
        Array of generated complex channel coefficients with shape (num_timestep, 2, num_device, num_subchannel + num_beam).
    """
    h_tilde = []
    for k in range(num_device):
        h_tilde_k_sub, h_tilde_k_mw = [], []
        for n in range(num_subchannel):
            h_tilde_k_sub.append(generate_h_tilde_device_channel(mu, sigma, num_timestep))
            
        for m in range(num_beam):
            h_tilde_k_mw.append(generate_h_tilde_device_channel(mu, sigma, num_timestep))

        h_tilde.append(np.array([h_tilde_k_sub, h_tilde_k_mw]))

    h_tilde = np.array(h_tilde) # [device index, interface, channel index, timestep]
    h_tilde = np.transpose(h_tilde, (3, 1, 0, 2)) # [timestep, interface, device index, channel_index] (3, 1, 0, 2)

    if save:
        with open(save_path, 'wb') as file:
            pickle.dump(h_tilde, file)

    return h_tilde


def compute_h_sub(distance_to_AP:float, h_tilde:np.ndarray) -> float:
    """
    Compute the channel coefficient for Sub-6GHz one connection.

    Parameters
    ----------
    distance_to_AP : float
        Distance to the AP in meters.
    h_tilde : np.ndarray
        Complex channel coefficient generated by `generate_h_tilde`.
    
    Returns
    -------
    h : np.ndarray
        Channel coefficient.
    """
    h = np.abs(h_tilde* pow(10, -path_loss_sub(distance=distance_to_AP)/20.0))**2

    return h

def compute_h_mW(distance_to_AP:float, is_blocked:bool, x:float, epsilon:float=0.005) -> float:
    """
    Compute the channel coefficient for mmWave one connection.
    
    Parameters
    ----------
    distance_to_AP : float
        Distance to the AP in meters.
    is_blocked : bool
        Whether the device is blocked by an obstacle.
    x : float
        Path loss variation.
    epsilon : float
        Side lobe beam gain.
    
    Returns
    -------
    h : float
        Channel coefficient.
    """
    
    # device blocked by obstacle
    if (is_blocked):
        path_loss = path_loss_mW_nlos(distance=distance_to_AP, x=x)
        h = G(epsilon=epsilon)*pow(10, -path_loss/10)*epsilon # G_Rx^k=epsilon
    # device not blocked
    else:
        path_loss = path_loss_mW_los(distance=distance_to_AP, x=x)
        h = G(epsilon=epsilon)**2*pow(10, -path_loss/10) # G_Rx^k = G_b

    return h


def signal_power(p:float, h:float) -> float:
    """
    Calculate the signal power.

    Parameters
    ----------
    p : float
        Transmit power.
    h : float
        Channel coefficient.

    Returns
    -------
    float
        Signal power.
    """
    return p * h


def gamma(w:float, s:float, interference:float, noise:float) -> Union[float, list[float, float]]:
    """
    Calculate the signal-to-noise ratio (SNR).
    
    Parameters
    ----------
    w : float 
        Bandwidth.
    s : float 
        Signal power.
    interference : float 
        Interference from other devices.
    noise : float 
        Noise power.
    return_power : bool
        Indicate whether to return power or not.
    
    Returns
    -------
    gamma : float
        SINR value if `return_power` is False.
    gamma, power : list of float
        [SINR, power] if `return_power` is True
    """
    interference_plus_noise = w*noise + interference
    gamma = s/interference_plus_noise

    return gamma


def compute_rate(w:float, sinr:float) -> float:
    """
    Calculate the achievable data rate.
    
    Parameters
    ----------
    w : float 
        Bandwidth.
    sinr : float
        Signal-to-Interference-plus-Noise Ratio (SINR).
    
    Returns
    -------
    rate : float
        rate in bits per second.
    """
    rate = w * np.log2(1 + sinr)

    return rate