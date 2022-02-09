import sys
import os
import glob
sys.path.append("/home/borlaff/GRID/")
import utils
from tqdm import tqdm
import tars
from astropy.io import fits
from astropy.wcs import WCS
import drizzlepac
import bottleneck as bn
#import ellipse
import numpy as np
import ds9tomask as dm
from bootmedian import bootmedian as bm
from drizzlepac import ablot
from stsci.tools import teal
from skimage import data
from skimage.restoration import inpaint
from astropy.modeling import models, fitting
import subprocess
from scipy.ndimage import gaussian_filter

xfin = 3000
yfin = 3000
ra_fin = 150.1126499
dec_fin = 2.3561308

def reset_mdrizsky(fits_list):
    for i in fits_list:
        print(i)
        reset_MDRIZSKY(i)


def create_deltaflat_files():
    F606W_dflat = fits.open("/home/borlaff/DF4/DELTAFLAT/final_acs-wfc_f606w_avg.fits")
    F814W_dflat = fits.open("/home/borlaff/DF4/DELTAFLAT/final_acs-wfc_f814w_avg.fits")
    F606W_dflat_sci1 = F606W_dflat[0].data[0:2048,:]
    F606W_dflat_sci2 = F606W_dflat[0].data[2048:,:]
    F814W_dflat_sci1 = F814W_dflat[0].data[0:2048,:]
    F814W_dflat_sci2 = F814W_dflat[0].data[2048:,:]

    F606W_dflat_sci1[np.where(F606W_dflat_sci1 < 0.3)] = 1
    F606W_dflat_sci1[np.where(F606W_dflat_sci1 > 1.7)] = 1

    F606W_dflat[0].data = gaussian_filter(F606W_dflat_sci1, sigma=3)
    F606W_dflat.verify("silentfix")
    os.system("rm /home/borlaff/DF4/DELTAFLAT/dflat_f606w_sci1.fits")
    F606W_dflat.writeto("/home/borlaff/DF4/DELTAFLAT/dflat_f606w_sci1.fits")

    F606W_dflat_sci2[np.where(F606W_dflat_sci2 < 0.3)] = 1
    F606W_dflat_sci2[np.where(F606W_dflat_sci2 < 1.7)] = 1

    F606W_dflat[0].data = gaussian_filter(F606W_dflat_sci2, sigma=3)
    F606W_dflat.verify("silentfix")
    os.system("rm /home/borlaff/DF4/DELTAFLAT/dflat_f606w_sci2.fits")
    F606W_dflat.writeto("/home/borlaff/DF4/DELTAFLAT/dflat_f606w_sci2.fits")

    F814W_dflat_sci1[np.where(F814W_dflat_sci1 < 0.3)] = 1
    F814W_dflat_sci1[np.where(F814W_dflat_sci1 < 1.7)] = 1
    F814W_dflat[0].data = gaussian_filter(F814W_dflat_sci1, sigma=3)
    F814W_dflat.verify("silentfix")
    os.system("rm /home/borlaff/DF4/DELTAFLAT/dflat_f814w_sci1.fits")
    F814W_dflat.writeto("/home/borlaff/DF4/DELTAFLAT/dflat_f814w_sci1.fits")

    F814W_dflat_sci2[np.where(F814W_dflat_sci2 < 0.3)] = 1
    F814W_dflat_sci2[np.where(F814W_dflat_sci2 < 1.7)] = 1
    F814W_dflat[0].data = gaussian_filter(F814W_dflat_sci2, sigma=3)
    F814W_dflat.verify("silentfix")
    os.system("rm /home/borlaff/DF4/DELTAFLAT/dflat_f814w_sci2.fits")
    F814W_dflat.writeto("/home/borlaff/DF4/DELTAFLAT/dflat_f814w_sci2.fits")
    print("Delta flats created!")


def launch_multidrizzle(subfix):
    HUDF_list = glob.glob(subfix)
    drizzlepac.astrodrizzle.AstroDrizzle(input=subfix, skysub=0,
                                        num_cores=1, clean=1,
                                        in_memory=0, preserve=0,
                                        combine_type='imedian', final_scale=0.06,
                                        final_kernel="lanczos3", driz_cr=1,
                                        updatewcs=0, final_pixfrac=1,
                                        final_outnx=xfin,final_outny=yfin,
                                        final_ra=ra_fin,final_rot=0,
                                        final_dec=dec_fin)

def run_astnoisechisel(fits_name, ext, tilesize=80):
    os.system("astnoisechisel -h" + str(ext) + " --tilesize="+str(tilesize)+","+str(tilesize)+" --interpnumngb=4 --keepinputdir --smoothwidth=15 " + fits_name)
    return(fits_name.replace(".fits", "_labeled.fits"))

def reset_bad_pixels(fits_name, ext):
    limit_down = -5000
    fits_file = fits.open(fits_name, mode = "update")
    fits_file[ext].data[(fits_file[ext].data < limit_down)] = 0
    fits_file.flush()
    fits_file.close()

def reset_MDRIZSKY(fits_name):
    fits_file = fits.open(fits_name, mode = "update")
    fits_file[1].header["MDRIZSKY"] = 0
    fits_file[4].header["MDRIZSKY"] = 0
    fits_file.flush()
    fits_file.close()

def remove_sky(fits_name, tilesize):
    # Make copy of flc
    os.system("cp " + fits_name + " " + fits_name.replace("_flc.fits", "_fld.fits"))
    output_name = fits_name.replace("_flc.fits","_sky.fits")
    fits_name = fits_name.replace("_flc.fits", "_fld.fits")

    reset_bad_pixels(fits_name,1)
    reset_bad_pixels(fits_name,4)
    # First extension
    labeled_name_1 = run_astnoisechisel(fits_name, 1, tilesize)
    labeled_fits = fits.open(labeled_name_1)
    fits_file = fits.open(fits_name)
    sky_gradient = labeled_fits[4].data
    print(labeled_fits[4].header)
    fits_file[1].data = fits_file[1].data - sky_gradient

    # Second extension
    labeled_name_2 = run_astnoisechisel(fits_name, 4)
    labeled_fits = fits.open(labeled_name_2)
    sky_gradient = labeled_fits[4].data
    print(labeled_fits[4].header)
    fits_file[4].data = fits_file[4].data - sky_gradient

    os.system("rm " + output_name)
    fits_file.verify("silentfix")
    fits_file.writeto(output_name)


def remove_gradient(input_flc, ext, detected_name):
    input_fits = fits.open(input_flc)
    detected_fits = fits.open(detected_name)
    input_fits[ext].data = detected_fits[1].data
    os.system("rm " + input_flc)
    input_fits.verify("silentfix")
    input_fits.writeto(input_flc)
    input_fits.close()
    detected_fits.close()


def mask_image(fits_list, ext=1, flat=True, dq_mask=True):

    # Check if astnoisechisel is installed:
    try:
        subprocess.call(["astnoisechisel"])
    except OSError as e:
        # handle file not found error.
        print("ERROR: Astnoisechisel not found!")
        os.kill(os.getpid(), signal.SIGUSR1)

    out_list = []
    if isinstance(fits_list, str):
        fits_list = [fits_list]
    if isinstance(fits_list, list):
        for i in range(len(fits_list)):
            fits_name = fits_list[i]

            fits_image_flatted = fits.open(fits_name)
            fits_image_flatted_name = fits_name.replace(".fits", "_flatted.fits")



            if flat:
                # flat_name = fits_image_flatted[0].header['PFLTFILE']
                flat_name = choose_master_flat(fits_name)
                flat_image = fits.open(flat_name)
                border = 5

                if (flat_image[ext].data.shape == (1024, 1024)) & (fits_image_flatted[ext].data.shape == (1014, 1014)):
                    flat_image_array = flat_image[ext].data[0+border:1024-border,
                                                      0 + border:1024-border]
                else:
                    flat_image_array = flat_image[ext].data
                fits_image_flatted[ext].data = np.divide(fits_image_flatted[ext].data, flat_image_array)

            hdul = fits.PrimaryHDU(fits_image_flatted[ext].data)
            hdul.verify("silentfix")
            os.system("rm "+fits_image_flatted_name)
            hdul.writeto(fits_image_flatted_name)


            print(fits_name)
            if fits_name[-6:] == "i.fits":
                out_name = fits_name.replace("i.fits", "m_ext" + str(ext) + ".fits")
            else:
                out_name = fits_name.replace(".fits", "_masked_ext" + str(ext) + ".fits")


            # Astnoisechisel
            cmd = "astnoisechisel -h0 --keepinputdir --tilesize=50,50 --interpnumngb=1 --smoothwidth=3 " + fits_image_flatted_name
            print(cmd)
            os.system(cmd)
            labeled_name = fits_image_flatted_name.replace(".fits", "_detected.fits")
            try:
                labeled_image = fits.open(labeled_name)
                fits_image = fits.open(fits_name)
                masked = fits_image[ext].data
                masked[(labeled_image[2].data !=0)] = np.nan

                if dq_mask:
                    masked[fits_image_flatted[3].data > 0] = np.nan

                hdu1 = fits.PrimaryHDU(masked)
                hdu1.verify("silentfix")
                os.system("rm "+out_name)
                hdu1.writeto(out_name)

                new_hdul = fits.open(out_name, "update")
                new_hdul[0].header = fits_image[ext].header
                new_hdul.verify("silentfix")
                new_hdul[0].header['SKYLVL'] = np.nanmedian(labeled_image[3].data)
                new_hdul[0].header['SKYSTD'] = np.nanmedian(labeled_image[4].data)
                new_hdul.flush()
                out_list.append(out_name)

            except IOError:
                print("Cannot run astnoisechisel! Probably too noisy image")
                return(["Error"])
    return(out_list)


def mask_and_crude_skycor_sci(input_sci, master_mask):
    mask_1 = mask_image(fits_list=[input_sci],ext=0, flat=False, dq_mask=False)
    print(mask_1)
    mask_1_fits = fits.open(mask_1[0], mode="update")
    mask_1_DF = fits.open(master_mask)
    mask_1_fits[0].data[mask_1_DF[0].data == 1] = np.nan
    mask_1_fits.flush()
    crude_skycor(input_sci, 0, mask_1[0], nsimul=1, noisechisel_grad=True, bootmedian=False)



def mask_and_crude_skycor(input_flc, mask_1_name, mask_2_name, nsimul=100):
    mask_1 = mask_image(fits_list=[input_flc],ext=1, flat=False)
    print(mask_1)
    mask_1_fits = fits.open(mask_1[0], mode="update")
    mask_1_DF = fits.open(mask_1_name)
    mask_1_fits[0].data[mask_1_DF[0].data == 1] = np.nan
    mask_1_fits.flush()
    crude_skycor(input_flc, 1, mask_1[0], nsimul=nsimul, noisechisel_grad=False, bootmedian=False)

    mask_4 = mask_image(fits_list=[input_flc],ext=4, flat=False)
    mask_4_fits = fits.open(mask_4[0], mode="update")
    mask_4_DF = fits.open(mask_2_name)
    mask_4_fits[0].data[mask_4_DF[0].data == 1] = np.nan
    mask_4_fits.flush()
    crude_skycor(input_flc, 4, mask_4[0], nsimul=nsimul, noisechisel_grad=False, bootmedian=False)


def copy_dq(input_flc, input_crcor):
    flc_fits = fits.open(input_flc, mode="update")
    crcor_fits = fits.open(input_crcor, mode="update")
    flc_fits[1].data = crcor_fits[1].data
    flc_fits[4].data = crcor_fits[4].data
    flc_fits[1].header = crcor_fits[1].header
    flc_fits[4].header = crcor_fits[4].header
    flc_fits.flush()

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
            fits_image[ext].data = fits_image[ext].data - skylvl["median"]
            fits_image[0].header['SKYSTD'] = skylvl["std1_down"]
            fits_image[0].header['SKYLVL'] = skylvl["median"]
            fits_image[ext].header['SKYSTD'] = skylvl["std1_down"]
            fits_image[ext].header['SKYLVL'] = skylvl["median"]
            os.system("rm " + fits_name)
            fits_image.verify("silentfix")
            fits_image.writeto(fits_name)
            fits_image.close()



def final_skycor(input_flc, ext, nsimul=25):
    mask_1 = mask_image(fits_list=[input_flc],ext=ext, flat=False, dq_mask = False)
    detected_name = input_flc.replace("_flc.fits", "_flc_flatted_detected.fits")
    remove_gradient(input_flc, ext, detected_name)
    crude_skycor(input_flc, ext, mask_1[0], nsimul, False)


def delta_flat_cor(input_flc, input_flat1, input_flat2):
    fits_file = fits.open(input_flc)
    dflat_file_sci1 = fits.open(input_flat1)
    dflat_file_sci2 = fits.open(input_flat2)
    dflat_file_sci1[0].data[np.isnan(dflat_file_sci1[0].data)]=1
    dflat_file_sci2[0].data[np.isnan(dflat_file_sci2[0].data)]=1
    dflat_file_sci1[0].data[np.where(dflat_file_sci1[0].data == 0)]=1
    dflat_file_sci2[0].data[np.where(dflat_file_sci2[0].data == 0)]=1
    fits_file[1].header["idcscale"] = 0.05
    fits_file[4].header["idcscale"] = 0.05
    fits_file[1].data = np.divide(fits_file[1].data,dflat_file_sci1[0].data)
    fits_file[4].data = np.divide(fits_file[4].data,dflat_file_sci2[0].data)
    fits_file.verify("silentfix")
    #outname = input_flc.replace("_flc.fits", "_flc.fits")
    outname=input_flc
    os.system("rm " + outname)
    fits_file.writeto(outname)


def get_parameters_list(fits_list, index, ext=0):
    PARAM = []
    for j in range(len(index)):
        PARAM.append([])
    for raw_name in fits_list:
        print(raw_name)
        raw_fits = fits.open(raw_name)
        for j in range(len(index)):
            try:
                PARAM[j].append(raw_fits[ext].header[index[j]])
            except KeyError:
                print("KeyError: Header keyword not found")
                PARAM[j].append("NONE")
    return(list(PARAM))


def separate_visits(fitslist):
    PARAMS = get_parameters_list(fitslist, ["ASN_ID"])
    unique_visits = np.array(list(set(PARAMS[0])))
    flocks = []
    fits_array = np.array(fitslist)
    for i in unique_visits:
        print(i)
        flock = fits_array[np.array(PARAMS[0]) == i]
        print(flock)
        flocks.append(list(flock))
    return(flocks)


def ensure_idcscale(input_flc):
    fits_file = fits.open(input_flc)
    fits_file[1].header["idcscale"] = 0.05
    fits_file[4].header["idcscale"] = 0.05
    fits_file.verify("silentfix")
    #outname = input_flc.replace("_flc.fits", "_flc_dflat.fits")
    outname=input_flc
    os.system("rm " + outname)
    fits_file.writeto(outname)



def astrometry_list(fitslist, refcat, rmax):

    #OLD SETUP, USING ASTRODRIZZLE's METHOD TO REMOVE CR
    for fitsimages in fitslist:
        rootname = os.path.basename(fitsimages).split("_")[0]
        os.system("rm " + rootname + "_crclean.fits")
        print(rootname)
    #chunks_final = separate_for_astrometry(fitslist, rmax)
    chunks_final = separate_visits(fitslist)

    for subset in chunks_final:

        if len(subset) > 20:
            step1, step2 = split_list(subset)
            drizzlepac.astrodrizzle.AstroDrizzle(input=step1, skysub=0,
                                                 static=0,
                                                 in_memory=1, preserve=0,
                                                 clean=0,
                                                 driz_sep_pixfrac=1.,
                                                 num_cores=1,
                                                 driz_sep_wcs=1,
                                                 driz_cr_corr=1,
                                                 resetbits=128,
                                                 driz_cr_snr='4.0 3.0',
                                                 driz_cr_scale='1.5 0.7',
                                                 driz_combine=0, updatewcs=0,
                                                 combine_type="imedian")

            drizzlepac.astrodrizzle.AstroDrizzle(input=step2, skysub=0,
                                                 static=0,
                                                 in_memory=1, preserve=0,
                                                 clean=0,
                                                 driz_sep_pixfrac=1.,
                                                 num_cores=1,
                                                 driz_sep_wcs=1,
                                                 driz_cr_corr=1,
                                                 resetbits=128,
                                                 driz_cr_snr='4.0 3.0',
                                                 driz_cr_scale='1.5 0.7',
                                                 driz_combine=0, updatewcs=0,
                                                 combine_type="imedian")
        else:

            drizzlepac.astrodrizzle.AstroDrizzle(input=subset, skysub=0,
                                                 static=0,
                                                 in_memory=1, preserve=0,
                                                 clean=0,
                                                 driz_sep_pixfrac=1.,
                                                 num_cores=1,
                                                 driz_sep_wcs=1,
                                                 driz_cr_corr=1,
                                                 resetbits=128,
                                                 driz_cr_snr='4.0 3.0',
                                                 driz_cr_scale='1.5 0.7',
                                                 driz_combine=0, updatewcs=0,
                                                 combine_type="imedian")






        cr_corrected_subset = []
        for i in subset:
            rootname = os.path.basename(i).split("_")[0]
            extname = os.path.basename(i).split("_")[1].split(".fits")[0]
            if extname == "flc":
                cr_name = rootname + "_crclean.fits"
            else:
                cr_name = rootname + "_" + extname + "_crclean.fits"
            print(cr_name)
            cr_corrected_subset.append(cr_name)
            ensure_dtype(cr_name)

            # Generamos los catalogos manualmente con SEXTRACTOR
            catalog_1_name = rootname + "_" + extname + "_crclean_sci1.cat"
            catalog_2_name = rootname + "_" + extname + "_crclean_sci2.cat"
            catfile_name = rootname + "_catfile.txt"

            os.system("sex " + cr_name + "[1] -c " + "/media/borlaff/CARGO/PHD/SEX_files/confi_tweakreg.sex " +
                      "-CHECKIMAGE_NAME image_segmentation.fits " +
                      "-CATALOG_NAME " + catalog_1_name)

            os.system("sex " + cr_name + "[4] -c " + "/media/borlaff/CARGO/PHD/SEX_files/confi_tweakreg.sex " +
                      "-CHECKIMAGE_NAME image_segmentation.fits " +
                      "-CATALOG_NAME " + catalog_2_name)


            os.system("echo " + cr_name + " " + catalog_1_name + " " + catalog_2_name + " > " + catfile_name)

            drizzlepac.tweakreg.TweakReg(files=cr_name,
                                         updatewcs=False, updatehdr=True,
                                         verbose=True, searchrad=5,
                                         searchunits="arcseconds",
                                         refcat=refcat, refxcol=4, refycol=5,
                                         catfile=catfile_name, xcol=2, ycol=3,
                                         interactive=0, use2dhist=True, see2dplot=True, fitgeometry="general")


def generate_manual_masks():
    for i in glob.glob("/home/borlaff/DF4/F*W/*sci1.reg"):
        dm.ds9tomask(fname=i, nx=4096, ny=2048, outname=i.replace(".reg","_mask.fits"))
    for i in glob.glob("/home/borlaff/DF4/F*W/*sci2.reg"):
        dm.ds9tomask(fname=i, nx=4096, ny=2048, outname=i.replace(".reg","_mask.fits"))



def generate_full_masks(fits_list):
    for input_flc in fits_list:
        mask_1 = mask_image(fits_list=[input_flc],ext=1, flat=False)
        print(mask_1)
        mask_1_fits = fits.open(mask_1[0])
        mask_1_DF = fits.open(input_flc.replace("_flc.fits","_") + "sci1_mask.fits")
        mask_1_fits[0].data[mask_1_DF[0].data == 1] = np.nan
        mask_1_fits.verify("silentfix")
        os.system("rm " + mask_1[0])
        mask_1_fits.writeto(mask_1[0])
        mask_1_fits.close()
        #crude_skycor(input_flc, 1, mask_1[0], nsimul=1, noisechisel_grad=False, bootmedian=False)

        mask_4 = mask_image(fits_list=[input_flc],ext=4, flat=False)
        mask_4_fits = fits.open(mask_4[0])
        mask_4_DF = fits.open(input_flc.replace("_flc.fits","_") + "sci2_mask.fits")
        mask_4_fits[0].data[mask_4_DF[0].data == 1] = np.nan
        mask_4_fits.verify("silentfix")
        os.system("rm " + mask_4[0])
        mask_4_fits.writeto(mask_4[0])
        mask_4_fits.close()



def remove_iref_list(filename_list, calibration_path):
    for filename in filename_list:
        remove_iref(filename, calibration_path)

def remove_iref(filename,calibration_path):
    raw_fits = fits.open(filename,ignore_missing_end=True)

    if(raw_fits[0].header['INSTRUME'] == 'WFC3'):

        if (raw_fits[0].header['BPIXTAB'][0:4]=="iref"):
            raw_fits[0].header['BPIXTAB']  = calibration_path + str.split(raw_fits[0].header['BPIXTAB'],"$")[-1]
            raw_fits[0].header['CCDTAB']   = calibration_path + str.split(raw_fits[0].header['CCDTAB'],"$")[-1]
            raw_fits[0].header['OSCNTAB']  = calibration_path + str.split(raw_fits[0].header['OSCNTAB'],"$")[-1]
            raw_fits[0].header['CRREJTAB'] = calibration_path + str.split(raw_fits[0].header['CRREJTAB'],"$")[-1]
            raw_fits[0].header['DARKFILE'] = calibration_path + str.split(raw_fits[0].header['DARKFILE'],"$")[-1]
            raw_fits[0].header['NLINFILE'] = calibration_path + str.split(raw_fits[0].header['NLINFILE'],"$")[-1]
            raw_fits[0].header['PFLTFILE'] = calibration_path + str.split(raw_fits[0].header['PFLTFILE'],"$")[-1]
            raw_fits[0].header['IMPHTTAB'] = calibration_path + str.split(raw_fits[0].header['IMPHTTAB'],"$")[-1]
            raw_fits[0].header['IDCTAB']   = calibration_path + str.split(raw_fits[0].header['IDCTAB'],"$")[-1]
            raw_fits[0].header['MDRIZTAB'] = calibration_path + str.split(raw_fits[0].header['MDRIZTAB'],"$")[-1]
        else:
            raw_fits[0].header['BPIXTAB']  = calibration_path + str.split(raw_fits[0].header['BPIXTAB'],"/")[-1]
            raw_fits[0].header['CCDTAB']   = calibration_path + str.split(raw_fits[0].header['CCDTAB'],"/")[-1]
            raw_fits[0].header['OSCNTAB']  = calibration_path + str.split(raw_fits[0].header['OSCNTAB'],"/")[-1]
            raw_fits[0].header['CRREJTAB'] = calibration_path + str.split(raw_fits[0].header['CRREJTAB'],"/")[-1]
            raw_fits[0].header['DARKFILE'] = calibration_path + str.split(raw_fits[0].header['DARKFILE'],"/")[-1]
            raw_fits[0].header['NLINFILE'] = calibration_path + str.split(raw_fits[0].header['NLINFILE'],"/")[-1]
            raw_fits[0].header['PFLTFILE'] = calibration_path + str.split(raw_fits[0].header['PFLTFILE'],"/")[-1]
            raw_fits[0].header['IMPHTTAB'] = calibration_path + str.split(raw_fits[0].header['IMPHTTAB'],"/")[-1]
            raw_fits[0].header['IDCTAB']   = calibration_path + str.split(raw_fits[0].header['IDCTAB'],"/")[-1]
            raw_fits[0].header['MDRIZTAB'] = calibration_path + str.split(raw_fits[0].header['MDRIZTAB'],"/")[-1]

        os.remove(filename)
        raw_fits.verify('silentfix')
        raw_fits.writeto(filename)
        raw_fits.close()

    if(raw_fits[0].header['INSTRUME'] == 'ACS'):
        for i in range(len(raw_fits)):
            for j in range(len(raw_fits[i].header)):
                if(isinstance(raw_fits[i].header[j], str)):
                    raw_fits[i].header[j] = raw_fits[i].header[j].replace("jref$",calibration_path)


        os.remove(filename)
        raw_fits.verify('silentfix')
        raw_fits.writeto(filename)
        raw_fits.close()


def mask_and_crude_skycor_DF4(fits_list):
    # F606W reset_MDRIZSKY to 0
    for i in glob.glob("/home/borlaff/DF4/F606W/*flc.fits"):
        print(i)
        mask_and_crude_skycor(i, i.replace("_flc.fits", "_sci1_mask.fits"), i.replace("_flc.fits", "_sci2_mask.fits"))





def ensure_dtype(fits_name):
    original_image = fits.open(fits_name)
    try:
        original_image[0].data = original_image[0].data.astype('>f4')
    except:
        print("No ext=0")
    try:
        original_image[1].data = original_image[1].data.astype('>f4')
    except:
        print("No ext=1")
    try:
        original_image[2].data = original_image[2].data.astype('>f4')
    except:
        print("No ext=2")
    try:
        original_image[3].data = original_image[3].data.astype('>i2')
    except:
        print("No ext=3")
    try:
        original_image[4].data = original_image[4].data.astype('>i2')
    except:
        print("No ext=4")
    try:
        original_image[5].data = original_image[5].data.astype('>f4')
    except:
        print("No ext=5")

    original_image.verify('silentfix')
    os.system("rm " + fits_name)
    original_image.writeto(fits_name)
    original_image.close()



def generate_stripe_model(masked_array):
    image_shape = masked_array.shape

    med_axis_0 = bn.nanmean(masked_array.astype("float32"), axis=0)
    med_axis_1 = bn.nanmean(masked_array.astype("float32"), axis=1)
    print(len(med_axis_0))
    stripe_model_axis_0 = np.zeros(image_shape)
    stripe_model_axis_1 = np.zeros(image_shape)

    for j in tqdm(range(image_shape[1])):
        stripe_model_axis_0[:,j] = med_axis_0[j]

    for i in tqdm(range(image_shape[0])):
        stripe_model_axis_1[i,:] = med_axis_1[i]


    stripe_model = (stripe_model_axis_0 + stripe_model_axis_1)/2.
    return(stripe_model)


def correct_amplifiers(masked_array):

    masked_array[:,0:2048] = masked_array[:,0:2048] - bn.nanmedian(masked_array[:,0:2048])
    masked_array[:,2048:]  = masked_array[:,2048:]  - bn.nanmedian(masked_array[:,2048:])

    return(masked_array)


def destripe_acs(flc_name, mask1_name, mask2_name):

    flc_fits = fits.open(flc_name)
    mask1_fits = fits.open(mask1_name)
    mask2_fits = fits.open(mask2_name)
    image_shape = flc_fits[1].data.shape

    flc_fits[1].data = correct_amplifiers(flc_fits[1].data.astype("float32"))
    flc_fits[4].data = correct_amplifiers(flc_fits[4].data.astype("float32"))

    masked_1_data = np.copy(flc_fits[1].data)
    masked_2_data = np.copy(flc_fits[4].data)
    masked_1_data[np.isnan(mask1_fits[0].data)] = np.nan
    masked_2_data[np.isnan(mask2_fits[0].data)] = np.nan
    stripe_model_1 = generate_stripe_model(masked_1_data)
    stripe_model_2 = generate_stripe_model(masked_2_data)

    stripe_model_1[np.isnan(stripe_model_1)] = 0.0
    stripe_model_2[np.isnan(stripe_model_2)] = 0.0

    flc_fits[1].data = flc_fits[1].data.astype("float32") - stripe_model_1
    flc_fits[4].data = flc_fits[4].data.astype("float32") - stripe_model_2

    os.system("rm " + flc_name)
    flc_fits.verify("silentfix")
    flc_fits.writeto(flc_name)

    mask1_fits[0].data = stripe_model_1
    mask2_fits[0].data = stripe_model_2
    os.system("rm stripe_model_1.fits stripe_model_2.fits")
    mask1_fits.verify("silentfix")
    mask2_fits.verify("silentfix")

    mask1_fits.writeto("stripe_model_1.fits")
    mask2_fits.writeto("stripe_model_2.fits")

    return(flc_name)
