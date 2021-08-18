import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy as ent

def entropy(hist, bit_instead_of_nat=False):
    """
    given a list of positive values as a histogram drawn from any information source,
    returns the entropy of its probability mass function. Usage example:
      hist = [513, 487] # we tossed a coin 1000 times and this is our histogram
      print entropy(hist, True)  # The result is approximately 1 bit
      hist = [-1, 10, 10]; hist = [0] # this kind of things will trigger the warning
    """
    # h = h[np.where(h!=0)[0]]
    h = np.asarray(hist, dtype=np.float64)
    if h.sum()<=0 or (h<0).any():
        print("[entropy] WARNING, malformed/empty input %s. Returning None."%str(hist))
        return None
    h = h/h.sum()
    log_fn = np.ma.log2 if bit_instead_of_nat else np.ma.log
    return -(h*log_fn(h)).sum()

def cross_entropy(x, y):
    """ Computes cross entropy between two distributions.
    Input: x: iterabale of N non-negative values
           y: iterabale of N non-negative values
    Returns: scalar
    """

    if np.any(x < 0) or np.any(y < 0):
        raise ValueError('Negative values exist.')

    # Force to proper probability mass function.
    x = np.array(x, dtype=np.float)
    y = np.array(y, dtype=np.float)
    x /= np.sum(x)
    y /= np.sum(y)

    # Ignore zero 'y' elements.
    mask = y > 0
    x = x[mask]
    y = y[mask]    
    ce = -np.sum(x * np.log(y)) 
    return ce

def generate_x(img_path):
    Ig = cv2.imread(img_path)
    # Ig = cv2.cvtColor(Ig, cv2.COLOR_BGR2GRAY)
    Ig = 0.299*Ig[:, :, 2] + 0.5870*Ig[:, :, 1] + 0.114*Ig[:, :, 0]
    Ig = Ig.astype('uint8')
    Ie = cv2.equalizeHist(Ig)
    ### Calculate ssim ###
    ssim_ig_ie, _ = ssim(Ig, Ie, full=True)

    ### Get histograms ###
    histg = cv2.calcHist([Ig],[0],None,[128],[0,256])
    histe = cv2.calcHist([Ie],[0],None,[128],[0,256])
    histg = np.reshape(histg, (histg.shape[0]))
    histe = np.reshape(histe, (histe.shape[0]))
    zero_idsg = np.where(histg==0)[0]
    zero_idse = np.where(histe==0)[0]
    zero_ids = np.unique(np.concatenate((zero_idsg, zero_idse)))
    histg = np.delete(histg, zero_ids)
    histe = np.delete(histe, zero_ids)

    ### Calculate entropies ###
    entropyg = entropy(histg)
    entropye = entropy(histe)

    ### Calculate cross_entropies ###
    cross_ent_ge = cross_entropy(histg, histe)
    cross_ent_eg = cross_entropy(histe, histg)
    # print(ssim_ig_ie, entropyg, entropye, cross_ent_ge, cross_ent_eg)
    return [ssim_ig_ie, entropyg, entropye, cross_ent_ge, cross_ent_eg]