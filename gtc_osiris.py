#######################################################
###                                                 ###
###     GTC SPECIFIC CLASS                       ###
###                                                 ###
#######################################################
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
from xmlr import xmliter, XMLParsingMethods
from tqdm import tqdm
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import WMAP9 as cosmo
from astropy.convolution import convolve, Gaussian2DKernel, Tophat2DKernel, convolve_fft
from astropy.time import Time
import box

def bias_correct(fits_list, bias_list):
    corrected_files = np.array([])
    for fits_name, bias_name in zip(fits_list, bias_list):
        if box.astheader(fits_name, 0, "BIASCOR")[0]=="False":
            fits_file = fits.open(fits_name)
            bias_file = fits.open(bias_name)
            fits_file[1].data = fits_file[1].data - bias_file[1].data
            fits_file[2].data = fits_file[2].data - bias_file[3].data
            fits_name_output = fits_name.replace(".fits","_b.fits")
            if os.path.exists(fits_name_output):
                os.remove(fits_name_output)
            fits_file.verify("silentfix")
            fits_file.writeto(fits_name_output)
            fits_file.close()
            bias_file.close()
            box.execute_cmd(cmd_text = "astfits -h0 " + fits_name_output + " --update=BIASCOR,True", verbose=False)
            box.execute_cmd(cmd_text = "astfits -h0 " + fits_name_output + " --update=BIASNAME," + bias_name, verbose=False)
            corrected_files = np.append(corrected_files, fits_name_output)
        else:
            corrected_files = np.append(corrected_files, fits_name)
    return(corrected_files)



def counts_to_flux(fits_list):
    corrected_files = np.array([])
    for fits_name in fits_list:

        unit_type_1 = box.astheader(fits_name, 1, "UNITS")
        unit_type_2 = box.astheader(fits_name, 2, "UNITS")

        if (unit_type_1 != unit_type_2):
            print("Error! Units in extension 1 != 2, Check files")
            return()

        if (unit_type_1 == ["flux"]):
            print("File " + fits_name + " already in flux units!")

        else:
           fits_file = fits.open(fits_name)
           exptime = fits_file[0].header["ELAPSHUT"]
           fits_file[1].data = fits_file[1].data/np.float32(exptime)
           fits_file[2].data = fits_file[2].data/np.float32(exptime)
           fits_name_output = fits_name
           if os.path.exists(fits_name_output):
               os.remove(fits_name_output)
           fits_file.verify("silentfix")
           fits_file.writeto(fits_name_output)
           fits_file.close()
           box.execute_cmd(cmd_text = "astfits -h0 " + fits_name_output + " --update=UNITS,flux", verbose=False)
           corrected_files = np.append(corrected_files, fits_name_output)
    return(corrected_files)



def flat_correct(fits_list, flat_list, th=0.8, dflat=False):
    corrected_files = np.array([])
    for fits_name, flat_name in zip(fits_list, flat_list):
        if box.astheader(fits_name, 0, "FLATCOR")[0]=="False":
            doflat=True
            subfix = "f.fits"
        elif box.astheader(fits_name, 0, "FLATCOR")[0]=="True" and dflat==True:
            doflat=True
            subfix = "df.fits"
        else:
            doflat=False

        if doflat:
            fits_file = fits.open(fits_name)
            flat_file = fits.open(flat_name)
            fits_file[1].data = fits_file[1].data/flat_file[1].data
            fits_file[2].data = fits_file[2].data/flat_file[3].data
            fits_file[1].data[np.where(flat_file[1].data<th)] = np.nan
            fits_file[2].data[np.where(flat_file[3].data<th)] = np.nan
            fits_name_output = fits_name.replace(".fits",subfix)
            if os.path.exists(fits_name_output):
                os.remove(fits_name_output)
            fits_file.verify("silentfix")
            fits_file.writeto(fits_name_output)
            fits_file.close()
            flat_file.close()

            if box.astheader(fits_name, 0, "FLATCOR")[0]=="False":
                box.execute_cmd(cmd_text = "astfits -h0 " + fits_name_output + " --update=FLATCOR,True", verbose=False)
                box.execute_cmd(cmd_text = "astfits -h0 " + fits_name_output + " --update=FLATNAME," + flat_name, verbose=False)
            if dflat:
                box.execute_cmd(cmd_text = "astfits -h0 " + fits_name_output + " --update=DFLATCOR,True", verbose=False)
                box.execute_cmd(cmd_text = "astfits -h0 " + fits_name_output + " --update=DFLATNAME," + flat_name, verbose=False)

            corrected_files = np.append(corrected_files, fits_name_output)
        else:
            corrected_files = np.append(corrected_files, fits_name)
    return(corrected_files)




def force_mask(input_fits, input_mask1, input_mask2):
    outname_list = []

    for fits_name, mask_name1, mask_name2 in tqdm(zip(input_fits, input_mask1, input_mask2)):

        fits_file = fits.open(fits_name)
        mask_file1 = fits.open(mask_name1)
        mask_file2 = fits.open(mask_name2)

        fits_file[1].data = fits_file[1].data.astype(np.float)
        fits_file[1].data[np.isnan(mask_file1[1].data)] = np.nan
        fits_file[2].data = fits_file[2].data.astype(np.float)
        fits_file[2].data[np.isnan(mask_file2[1].data)] = np.nan

        fits_file.verify("silentfix")
        outname = fits_name.replace(".fits", "m.fits")
        os.system("rm " + outname)
        fits_file.writeto(outname)
        outname_list.append(outname)
    return(np.array(outname_list))


def stack_multifits(fits_target, fits_input, ext):
    if not isinstance(ext, list):
        ext = [ext]
    #if not isinstance(ext_target, list):
    #    ext_target = [ext_target]
    #for i,j in zip(ext, ext_target):
    for i in ext:
        cmd_text = "astfits " + fits_input + " -h" + str(i) + " --copy=" + str(i) + " --output=" + fits_target
        box.execute_cmd(cmd_text, verbose=True)

    box.execute_cmd(cmd_text = "astfits -h0 " + fits_target + " --write=EXTNAME,INFO", verbose=True)
    box.execute_cmd(cmd_text = "astfits -h1 " + fits_target + " --write=EXTNAME,BIAS_CCD1", verbose=True)
    box.execute_cmd(cmd_text = "astfits -h2 " + fits_target + " --write=EXTNAME,STD_BIAS_CCD1", verbose=True)
    box.execute_cmd(cmd_text = "astfits -h3 " + fits_target + " --write=EXTNAME,BIAS_CCD2", verbose=True)
    box.execute_cmd(cmd_text = "astfits -h4 " + fits_target + " --write=EXTNAME,STD_BIAS_CCD2", verbose=True)
    return(fits_target)


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
