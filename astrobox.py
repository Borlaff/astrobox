# Astro Box: Astronomy Imaging Tool Box
# Alejandro Serrano Borlaff - NASA Postdoctoral Fellow
# v1.0 - 18 Mar 2020: Assimilating Coop, Utils DF4, Utils GTC, Tars
#
# -------------------------------------------------- #
import os
import sys
import glob
import subprocess
import datetime
import healpy as hp
import numpy as np
import pandas as pd
import bottleneck as bn
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
#import gizmo_read
import shlex
import multiprocessing
from reproject import reproject_interp
# from xmlr import xmliter, XMLParsingMethods
from tqdm import tqdm
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import WMAP9 as cosmo
from astropy.convolution import convolve, Gaussian2DKernel, Tophat2DKernel, convolve_fft
from astropy.time import Time

import euclid
import gtc_osiris
import irsa
import box
import profiles
import bootmedian

print("Astrobox v.1.0")
print("Astronomy Imaging Tool Box - A box of usual tools for astronomical imaging processing")
print("Author: Alejandro S. Borlaff - NASA Ames Research Center - a.s.borlaff@nasa.gov / asborlaff@gmail.com")
