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
import gizmo_read
import shlex
import multiprocessing
from reproject import reproject_interp
from xmlr import xmliter, XMLParsingMethods
from tqdm import tqdm
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import WMAP9 as cosmo
from astropy.convolution import convolve, Gaussian2DKernel, Tophat2DKernel, convolve_fft
from astropy.time import Time

from euclid import euclid as euclid
from irsa import irsa as irsa
from bootima import bootima as bm


def execute_cmd(cmd_text, verbose=False):
    """
    execute_cmd: This program executes in shell the input string command, printing the output for the user
    """
    print(cmd_text)
    with open('temp.log', 'wb') as f:
        process = subprocess.Popen(cmd_text, stdout=subprocess.PIPE, shell=True)
        for line in iter(process.stdout.readline, b''):  # With Python 3, you need iter  (process.stdout.readline, b'') (i.e. the sentinel passed to iter needs to be a binary string, since b'' != '')
            sys.stdout.write(line.decode(sys.stdout.encoding))
            f.write(line)

    #out, error = p.communicate()
    #if ((len(error) > 5) and verbose):
    #    print(error)



def smooth_fits(fits_file, ext, sigma):
    """
    This program applies a gaussian smoothing of s=SIGMA to the array stored in the extension EXT of the FITS_FILE.
    """
    fits_image = fits.open(fits_image)
    fits_image[ext].data = gaussian_filter(fits_image[ext].data, sigma=sigma)
    os.system("rm " + fits_file)
    fits_image.verify("silentfix")
    fits_image.writeto(fits_file)
    fits_image.close()
    return(fits_file)



def generate_flag_ima(fits_list, ext):
    """
    This program generates a flag image for Sextractor, which sets to 1 (flagged) all pixels with a value of 0 or NAN
    To be done: accept a different kind of mask as input - I.E: Noisechisel detection map
    """
    if isinstance(fits_list, str):
        fits_list = [fits_list]

    flag_name_output = []
    for i in fits_list:
        flag_name = i.replace(".fits", "_flag.fits")
        input_fits = fits.open(i)
        flag_ima = np.zeros(input_fits[ext].data.shape).astype("int")
        flag_ima[:] = int(0)
        flag_ima[np.isnan(input_fits[ext].data)] = int(1)
        flag_ima[input_fits[ext].data==0] = int(1)
        os.system("rm " + flag_name)
        hdu = fits.PrimaryHDU(flag_ima)
        hdu.scale('int32')
        hdul = fits.HDUList([hdu])
        os.system("rm " + flag_name)
        hdul.writeto(flag_name)
        flag_name_output.append(flag_name)
    return(flag_name_output)



def astheader(fits_list, ext, key):
    """
    astheader: Writes keywords on fits files using astfits from GnuAstro
    """

    # First we check if the input is a list:
    if not isinstance(fits_list, (list,np.ndarray)):
        fits_list = [fits_list]

    output_list = []
    for fits_file in fits_list:
        cmd_text = "astfits -h" + str(ext) + " " + fits_file + " | grep '" + key + "' | awk '{print $3}'"
        #print(cmd_text)
        p = subprocess.Popen(cmd_text, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        out, error = p.communicate()
        try:
            output = float(out)
        except ValueError:
            output = out.decode("utf-8").replace("'" , "").replace("\n" , "")
        output_list.append(output)
    return(output_list)


def interpolate_array(data):
    print("Interpolating NaNs")
    interp1 = np.array(pd.DataFrame(data).interpolate(axis=0, limit_direction="both"))
    interp1 = np.array(pd.DataFrame(interp1).interpolate(axis=1, limit_direction="both"))
    interp2 = np.array(pd.DataFrame(data).interpolate(axis=1, limit_direction="both"))
    interp2 = np.array(pd.DataFrame(interp2).interpolate(axis=0, limit_direction="both"))
    interp = (interp1 + interp2)/2.
    return(interp)



def zeros_to_nan(fits_name, ext):
    print("Remove zeros from " + fits_name)
    fits_file = fits.open(fits_name, mode="update")
    fits_file[ext].data[fits_file[ext].data == 0] = np.nan
    fits_file.flush()
    fits_file.close()


def normalize_frame(fits_list, ext):
    """
    normalize_frame: Normalize the data extension from the input list of files
    """
    if not isinstance(fits_list, list):
        fits_list=[fits_list]
    arguments = zip(np.array(fits_list), np.array([ext]*len(fits_list)))

    nproc = multiprocessing.cpu_count() - 2
    pool = multiprocessing.Pool(processes=nproc)

    print("Starting pool paralel normalize")
    for _ in tqdm(pool.starmap(normalize_single_frame, arguments), total=len(fits_list)):
        pass
    pool.terminate()


def normalize_single_frame(fits_list, ext):
    """
    Normalize the data extension from the input list of files
    """
    corrected_files = np.array([])
    if not isinstance(fits_list, list):
        fits_list = [fits_list]


    for fits_name in fits_list:
        fits_file = fits.open(fits_name)

        norma = bn.nanmedian(fits_file[ext].data)
        #if ext==1:
        #    norma = bn.nanmedian(fits_file[ext].data[850:1000,850:1000])
        #if ext==2:
        #    norma = bn.nanmedian(fits_file[ext].data[0:250,850:1000])

        fits_file[ext].data = fits_file[ext].data/norma
        if os.path.exists(fits_name):
            os.remove(fits_name)
        fits_file.verify("silentfix")
        fits_file.writeto(fits_name)
        fits_file.close()
        execute_cmd(cmd_text = "astfits -h" + str(ext)+ " " + fits_name + " --update=NORMAL,True", verbose=False)
        execute_cmd(cmd_text = "astfits -h" + str(ext)+ " " + fits_name + " --update=NORMFACT," + str(np.round(norma,8)), verbose=False)
        corrected_files = np.append(corrected_files, fits_name)
    return(corrected_files)





def mask_fits(fits_list, ext, clean=True):
    output_list = []

    if not isinstance(fits_list, (list, np.ndarray)):
        fits_list=[fits_list]

    if isinstance(ext, int):
        ext=[ext]*len(fits_list)

    for i,j in tqdm(zip(fits_list, ext)):
        file_name = os.path.abspath(i)
        outname = i.replace(".fits", "m" + str(j) + ".fits")

        if os.path.isfile(outname):
            os.remove(outname)
        #execute_cmd(cmd_text="cp " + file_name + " " + outname)
        # Ensure everything is clean
        # Cleaning everything for the next loop
        os.system("rm dummy.fits")
        os.system("rm masked.fits")
        os.system("rm detections_mask.fits")
        os.system("rm detections.fits")

        # ##################################################################################
        # Hay que interpolar los CR antes de enmascarar, o perdemos demasiado area #########
        ####################################################################################
        cmd_text = "astfits -h" + str(j) + " " + file_name + " --copy=" + str(j) + " --output=dummy.fits"
        execute_cmd(cmd_text, verbose=True)

        cmd_text = "astnoisechisel --keepinputdir dummy.fits -h1 --output=detections.fits"
        execute_cmd(cmd_text, verbose=True)
        # Mask to 1 the j + 2 extension when is detected and dq extension is 0
        cmd_text = "astarithmetic --keepinputdir -h2 detections.fits 0 gt --output=detections_mask.fits"
        execute_cmd(cmd_text, verbose=True)
        cmd_text = "astarithmetic --keepinputdir -h" + str(j) + " " + file_name + " -h1 detections_mask.fits nan where --output=masked.fits"
        execute_cmd(cmd_text, verbose=True)

        cmd_text = "astfits -h1 masked.fits --copy=1 --output=" + outname
        execute_cmd(cmd_text, verbose=True)

        # Cleaning everything for the next loop
        if clean:
            os.system("rm dummy.fits")
            os.system("rm masked.fits")
            os.system("rm detections_mask.fits")
            os.system("rm detections.fits")
        output_list.append(outname)
    return(output_list)



def GTC_IRAF_distortion(fits_list):
    """
    This program executes the mosaic_2x2_v2.cl map from python to IRAF using shell.
    """
    if isinstance(fits_list, str):
        fits_list = [fits_list]

    for i in fits_list:
        os.system("cp -v /home/borlaff/GTC/IRAF/mosaic_2x2_v2.cl .")
        os.system("echo task \$mosaic=mosaic_2x2_v2.cl > iraf_line_1.txt")
        os.system("echo mosaic " + i + " > iraf_line_2.txt")
        os.system("echo logout > iraf_line_3.txt")
        os.system("cat iraf_line_1.txt iraf_line_2.txt iraf_line_3.txt > iraf.input")
        os.system("cl < iraf.input")
        os.system("mv OsirisMosaic.fits " + i.replace(".fits", "_drc.fits"))
