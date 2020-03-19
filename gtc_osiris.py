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


def zeros_to_nan(fits_list, ext):
    corrected_files = np.array([])
    for fits_name in fits_list:
        fits_file = fits.open(fits_name)
        fits_file[ext].data[fits_file[ext].data == 0] = np.nan
        fits_name_output = fits_name
        if os.path.exists(fits_name_output):
            os.remove(fits_name_output)
        fits_file.verify("silentfix")
        fits_file.writeto(fits_name_output)
        fits_file.close()
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


def normalize_frame(fits_list, ext):
    corrected_files = np.array([])
    for fits_name in fits_list:
        fits_file = fits.open(fits_name)

        if ext==1:
            norma = bn.nanmedian(fits_file[ext].data[600:1400,200:800])
        if ext==2:
            norma = bn.nanmedian(fits_file[ext].data[600:1400,100:700])

        fits_file[ext].data = fits_file[ext].data/norma
        if os.path.exists(fits_name):
            os.remove(fits_name)
        fits_file.verify("silentfix")
        fits_file.writeto(fits_name)
        fits_file.close()
        box.execute_cmd(cmd_text = "astfits -h" + str(ext)+ " " + fits_name + " --update=NORMAL,True", verbose=False)
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
        box.execute_cmd(cmd_text, verbose=False)

    box.execute_cmd(cmd_text = "astfits -h0 " + fits_target + " --write=EXTNAME,INFO", verbose=False)
    box.execute_cmd(cmd_text = "astfits -h1 " + fits_target + " --write=EXTNAME,BIAS_CCD1", verbose=False)
    box.execute_cmd(cmd_text = "astfits -h2 " + fits_target + " --write=EXTNAME,STD_BIAS_CCD1", verbose=False)
    box.execute_cmd(cmd_text = "astfits -h3 " + fits_target + " --write=EXTNAME,BIAS_CCD2", verbose=False)
    box.execute_cmd(cmd_text = "astfits -h4 " + fits_target + " --write=EXTNAME,STD_BIAS_CCD2", verbose=False)
    return(fits_target)


def final_skycor(input_flc, ext, nsimul=25):
    mask_1 = box.mask_image(fits_list=[input_flc], ext=ext, flat=False, dq_mask = False)
    detected_name = input_flc.replace("_flc.fits", "_flc_flatted_detected.fits")
    #remove_gradient(input_flc, ext, detected_name)
    crude_skycor(input_flc, ext, mask_1[0], nsimul, False, True)


def crude_skycor(fitslist, ext, mask=None, nsimul=100, noisechisel_grad=False, bootmedian=True):
    if isinstance(fitslist, str):
        fitslist = [fitslist]
    if isinstance(fitslist, list):
        for fits_name in fitslist:
            print(fits_name)
            fits_image = fits.open(fits_name)

            if mask is not None:
                print("Input mask accepted: " + mask)
                mask_fits = fits.open(mask)
                shape_mask = mask_fits[0].data.shape
                shape_fits = fits_image[ext].data.shape
                mask_array = mask_fits[0].data
                if (shape_mask == (1014, 1014)) & (shape_fits == (1024, 1024)):
                    mask_array = np.zeros(shape_fits)
                    border = 5
                    mask_array[0+border:1024-border,
                               0+border:1024-border] = mask_fits[0].data
                if bootmedian:
                    skylvl = bm.bootmedian(sample_input=fits_image[ext].data[~np.isnan(mask_array)],
                                           nsimul=nsimul, errors=1)
                if not bootmedian:
                    median_sky = bn.nanmedian(fits_image[ext].data[~np.isnan(mask_array)])
                    #sigma_sky = bn.nanstd(fits_image[ext].data[~np.isnan(mask_array)])
                    sigma_sky = 0
                    skylvl = {"median": median_sky,
                              "s1_up": median_sky+sigma_sky,
                              "s1_down": median_sky-sigma_sky,
                              "std1_down": sigma_sky,
                              "std1_up": sigma_sky}
            else:
                if bootmedian:
                    skylvl = bm.bootmedian(sample_input=fits_image[ext].data, nsimul=nsimul, errors=1)
                if not bootmedian:
                    median_sky = bn.nanmedian(fits_image[ext].data)
                    #sigma_sky = bn.nanstd(fits_image[ext].data)
                    sigma_sky = 0

                    skylvl = {"median": median_sky,
                              "s1_up": median_sky+sigma_sky,
                              "s1_down": median_sky-sigma_sky,
                              "std1_down": sigma_sky,
                              "std1_up": sigma_sky}

            print(skylvl)
            print(np.abs(skylvl["median"] - skylvl["s1_up"])/2.)
            print("Skylvl: " + str(skylvl["median"]) + " +/- " + str(np.abs(skylvl["s1_up"] - skylvl["s1_down"])/2.))
            fits_image[ext].data = np.float32(fits_image[ext].data - skylvl["median"])
            fits_image[0].header['SKYSTD'] = skylvl["std1_down"]
            fits_image[0].header['SKYLVL'] = skylvl["median"]
            fits_image[ext].header['SKYSTD'] = skylvl["std1_down"]
            fits_image[ext].header['SKYLVL'] = skylvl["median"]
            os.system("rm " + fits_name)
            fits_image.verify("silentfix")
            fits_image.writeto(fits_name)
            fits_image.close()
