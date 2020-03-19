# Alejandro Serrano Borlaff
# NASA Postdoctoral Fellow - 1 December 2019
# NASA Ames Research Center - 94041 Moffett Field, Mountain View, CA, USA.
#
# Contact:
# a.s.borlaff@nasa.gov
# asborlaff@gmail.com
#
#    Bootima
#    A program to calculate median of large amount of images stored in fits files.
#
#    v2.0: First working version
#    v2.1: Starting to add support for more images than RAM through chunking
##############################################################################

import os
import sys
import glob
import psutil
import miniutils
import subprocess
import numpy as np
import pandas as pd
import bottleneck as bn
import multiprocessing
from tqdm import tqdm
from astropy.io import fits

# results = parallel_progbar(do_something_slow, my_list)
# Equivalent to a parallel version of [do_something_slow(x) for x in my_list]


# DO TO:
# 2 - Avoid creating slices again if they are already in place.

def does_it_fit_in_RAM(fits_list, ext):
    # Quite slow so far, we must paralelize this.
    if not isinstance(fits_list, list):
        fits_list=[fits_list]

    nimages = len(fits_list)

    NAXIS1=[]
    NAXIS2=[]
    bitpix=[]

    print("Getting NAXIS1")
    nprocs = multiprocessing.cpu_count() - 2
    arguments = zip(np.array(fits_list), np.array([ext]*len(fits_list)), np.array(["NAXIS1"]*len(fits_list)))
    #NAXIS1_out = miniutils.parallel_progbar(astheader, arguments)
    NAXIS1_out = miniutils.parallel_progbar(astheader, arguments, starmap=True, nprocs=nprocs)
    #NAXIS1_out = list(tqdm(pool.starmap(astheader, arguments), total=len(fits_list)))
    NAXIS1 = [i[0] for i in NAXIS1_out]

    print("Getting NAXIS2")
    arguments = zip(np.array(fits_list), np.array([ext]*len(fits_list)), np.array(["NAXIS2"]*len(fits_list)))
    NAXIS2_out = miniutils.parallel_progbar(astheader, arguments, starmap=True, nprocs=nprocs)
    NAXIS2 = [i[0] for i in NAXIS2_out]

    print("Getting BITPIX")
    arguments = zip(np.array(fits_list), np.array([ext]*len(fits_list)), np.array(["BITPIX"]*len(fits_list)))
    bitpix_out = miniutils.parallel_progbar(astheader, arguments, starmap=True, nprocs=nprocs)
    bitpix = [i[0] for i in bitpix_out]



    print("Checking if dataset fits in RAM:")

    print(NAXIS1)

    #for i in tqdm(fits_list):
    #    NAXIS1.append(astheader(i, ext, "NAXIS1")[0])
    #    NAXIS2.append(astheader(i, ext, "NAXIS2")[0])
    #    bitpix.append(astheader(i, ext, "BITPIX")[0])

    if not (len(set(NAXIS1))==1) and (len(set(NAXIS2))==1):
        print("Error: All images must have the same dimensions")

    NAXIS1 = int(NAXIS1[0])
    NAXIS2 = int(NAXIS2[0])
    bitpix = np.abs(int(bitpix[0]))

    # Second: check size of files and size of RAM
    image_size = bitpix*np.array(NAXIS1)*np.array(NAXIS2)/8/1024/1024 # In Mb
    dataset_size = image_size*nimages
    print("Image dimensions: " + str(NAXIS1) + " x " + str(NAXIS2) + " x " + str(nimages) + " px³")
    print("Bitpix: " + str(bitpix) + " bits per pix")
    print("Image extension size: " + str(image_size) + " Mb per ext")
    # What is the size of the complete cube?
    print("Full dataset size: " + str(dataset_size/1024) + " Gb")
    # Anaylize system settings, max RAM of system
    available_memory = psutil.virtual_memory().available/1024/1024 # In Mb
    print("Available RAM memory: " + str(available_memory/1024) + " Gb")

    if available_memory > dataset_size:
        print("----------------------------------")
        print("No slicing needed - Running standard Bootima")
        print("----------------------------------")
        return([True, {"NAXIS1": NAXIS1, "NAXIS2": NAXIS2, "available_memory": available_memory,
                        "dataset_size": dataset_size, "nimages": nimages, "image_size": image_size, "bixpix": bitpix}])
    else:
        print("----------------------------------")
        print("Not enough memory for full bootima")
        print("----------------------------------")
        print("----------> Running slice analysis")
        return([False, {"NAXIS1": NAXIS1, "NAXIS2": NAXIS2, "available_memory": available_memory,
                        "dataset_size": dataset_size, "nimages": nimages, "image_size": image_size, "bixpix": bitpix}])


def execute_cmd(cmd_text, verbose=False):
    """
    This program executes in shell the input string command, printing the output for the user
    """
    if verbose:
        print(cmd_text)
    with open('temp.log', 'wb') as f:
        process = subprocess.Popen(cmd_text, stdout=subprocess.PIPE, shell=True)
        for line in iter(process.stdout.readline, b''):  # With Python 3, you need iter  (process.stdout.readline, b'') (i.e. the sentinel passed to iter needs to be a binary string, since b'' != '')
            sys.stdout.write(line.decode(sys.stdout.encoding))
            f.write(line)

    #out, error = p.communicate()
    #if ((len(error) > 5) and verbose):
    #    print(error)


def astheader(fits_list, ext, key, verbose=False):
    """
    Writes keywords on fits files using astfits from GnuAstro
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



def bootima_slice(fits_list, ext, nsimul, outname, clean=True, verbose=False, mode="median"):
    """
    A program to calculate median of large amount of images stored in fits files.

    v2.0: First working version
    v2.1: Starting to add support for more images than RAM through chunking
    """


    if os.path.isabs(outname):
        output_dir = os.path.dirname(outname)
        median_output_name = outname
        #std_output_name = outname.replace(".fits","_std.fits")
        s1up_output_name = outname.replace(".fits","_s1up.fits")
        s1down_output_name = outname.replace(".fits","_s1down.fits")
        s1mean_output_name = outname.replace(".fits","_s1mean.fits")

    else:
        output_dir = os.path.dirname(os.path.abspath(fits_list[0]))
        median_output_name = output_dir + "/" + outname
        #std_output_name = output_dir + "/" + outname.replace(".fits","_std.fits")
        s1up_output_name = output_dir + "/" + outname.replace(".fits","_s1up.fits")
        s1down_output_name = output_dir + "/" + outname.replace(".fits","_s1down.fits")
        s1mean_output_name = output_dir + "/" + outname.replace(".fits","_s1mean.fits")

    simul_names = []
    basename_simul_names = []

    if isinstance(ext, int):
        ext = [ext]*len(fits_list)

    # We move to the files directory
    # This is a bad practice, but it is the only way to shorten the input line
    current_wd = os.getcwd()
    fits_wd = os.path.dirname(os.path.abspath(fits_list[0]))
    os.chdir(fits_wd)
    basename_fits_list = [os.path.basename(i) for i in fits_list]


    for i in tqdm(range(0,nsimul)):
        if verbose:
            print(i)
        boot_indexes = np.random.randint(0,len(basename_fits_list), len(basename_fits_list))
        str_image_list = ""
        str_ext_list = ""
        sim_output_name = output_dir + "/simul_" + str(i).zfill(7) + ".fits"
        simul_names.append(sim_output_name)
        basename_simul_names.append("simul_" + str(i).zfill(7) + ".fits")

        for j in boot_indexes:
            str_image_list = str_image_list + " " + basename_fits_list[j]
            if len(set(ext)) != 1:
                str_ext_list = str_ext_list + " -h" + str(ext[j])

        if len(set(ext)) == 1:
            str_ext_list = " -g" + str(ext[0])
            # astarithmetic -h1 masked_ext1.fits -h2 masked_ext1.fits 0 gt nan where
        cmd_text = "astarithmetic --keepinputdir " + str_image_list + " " + str_ext_list + " " + str(len(boot_indexes)) + " " + mode + " --output=" + sim_output_name
        execute_cmd(cmd_text, verbose=verbose)

    str_simulation_list = ""
    for m in simul_names:
        str_simulation_list = str_simulation_list + " " + m

    basename_str_simulation_list = ""
    for m in basename_simul_names:
        basename_str_simulation_list = basename_str_simulation_list + " " + m

    # We calculate the median image

    cmd_text = "astarithmetic --keepinputdir " + basename_str_simulation_list + " -g1 " + str(nsimul) + " " + mode + " --output=" + median_output_name
    execute_cmd(cmd_text, verbose=verbose)

    # We calculate the standard deviation image
    #cmd_text = "astarithmetic --keepinputdir " + basename_str_simulation_list + " -g1 " + str(nsimul) + " std --output=" + std_output_name
    #execute_cmd(cmd_text, verbose=verbose)
    cmd_text = "astarithmetic --keepinputdir " + basename_str_simulation_list + " -g1 " + str(nsimul) + " 0.1586553 quantile --output=" + s1down_output_name
    execute_cmd(cmd_text, verbose=verbose)

    cmd_text = "astarithmetic --keepinputdir " + basename_str_simulation_list + " -g1 " + str(nsimul) + " 0.8413447 quantile --output=" + s1up_output_name
    execute_cmd(cmd_text, verbose=verbose)

    str_s1_error_list = s1up_output_name  + " " + s1down_output_name
    cmd_text = "astarithmetic --keepinputdir  -g1 " + str_s1_error_list + " - 2 / --output=" + s1mean_output_name
    execute_cmd(cmd_text, verbose=verbose)


    # We move back to the previous wd
    os.chdir(current_wd)

    # We join the two output files into one single file.
    # Extension 0: Empty
    # Extension 1: Median image
    # Extension 2: Standard deviation

    cmd_text = "astfits " + s1mean_output_name + " -h1 --copy 1 --output=" + median_output_name
    execute_cmd(cmd_text, verbose=verbose)
    execute_cmd(cmd_text = "astfits -h0 " + median_output_name + " --update=EXTNAME,INFO", verbose=verbose)
    execute_cmd(cmd_text = "astfits -h1 " + median_output_name + " --update=EXTNAME,MEDIAN", verbose=verbose)
    execute_cmd(cmd_text = "astfits -h2 " + median_output_name + " --update=EXTNAME,STD", verbose=verbose)
    if (clean and os.path.exists(median_output_name)):
        print("Cleaning temp files...")
        for k in simul_names:
            if os.path.exists(k):
                os.remove(k)

        if os.path.exists(s1up_output_name):
            os.remove(s1up_output_name)
        if os.path.exists(s1down_output_name):
            os.remove(s1down_output_name)
        if os.path.exists(s1mean_output_name):
            os.remove(s1mean_output_name)


    else:
        print("Median output not created: Check log.")
        print("Leaving simulations for reprocessing if required.")


def slice_fits(fits_list, ext, nslices):

    nzfill = int(np.ceil(np.log10(nslices+1)))
    if not isinstance(fits_list, (list,np.ndarray)):
        fits_list = [fits_list]

    output = []

    # Here we have to calculate the sizes of the slices.
    # Take into account that the number of pixels might odd
    # That the slices can be odd too, and thus it has to be calculated while slicing.
    print("Slicing fits")
    for fits_name in fits_list:
        # For each image
        outslices = []
        slice_list = list(np.linspace(0, nslices-1, nslices, dtype="int"))
        last_slice = slice_list[-1]
        NAXIS1 = int(astheader(fits_name, ext, "NAXIS1")[0])
        NAXIS2 = int(astheader(fits_name, ext, "NAXIS2")[0])
        minx=1
        slice_size=int(NAXIS1/nslices)
        maxx=slice_size
        print("Slice list " + str(slice_list))
        for i in slice_list:
            slice_name = fits_name.replace(".fits","_s" + str(i).zfill(nzfill) + ".fits")
            if os.path.exists(slice_name):
                print("Slice " +  slice_name + " found! - Jumping")
                outslices.append(slice_name)
            else:
            # For each slice
                if i == last_slice:
                    maxx = NAXIS1
                print("Slice " + str(i) + ": " + str(minx) + " - " + str(maxx))
                cmd = "astcrop --mode=img -h" + str(ext) + " --section='" + str(minx) + ":" + str(maxx) + \
                      ",1:" + str(NAXIS2) + "' -o " + slice_name + " " + fits_name
                execute_cmd(cmd)
                minx = maxx+1
                maxx = maxx + slice_size
                outslices.append(slice_name)
        output.append(outslices)
    return(np.array(output))


def reconstruct_slices(slice_list, ext, outname):
    slice_list.sort()
    print(slice_list)
    nimages = len(slice_list)
    NAXIS1=[]
    NAXIS2=[]
    bitpix=[]

    for i in slice_list:
        NAXIS1.append(int(astheader(i, ext, "NAXIS1")[0]))
        NAXIS2.append(int(astheader(i, ext, "NAXIS2")[0]))
        bitpix.append(int(astheader(i, ext, "BITPIX")[0]))

    canvas = np.zeros((NAXIS2[0],np.sum(np.array(NAXIS1))))

    minx = 0
    maxx = 0
    miny = 0
    maxy = NAXIS2[::-1][0]

    for slice_name in slice_list:
        slice_fits = fits.open(slice_name)
        slice_shape = slice_fits[ext].data.shape
        maxx = maxx + slice_shape[1]

        print("NAXIS1: " + str(NAXIS1))
        print("NAXIS2: " + str(NAXIS2))
        print("canvas[" + str(miny) + ":" + str(maxy) + "," + str(minx) + ":" + str(maxx) + "]")
        canvas[miny:maxy,minx:maxx] = slice_fits[ext].data
        minx = minx + slice_shape[1]

    hdu1 = fits.PrimaryHDU()
    hdu2 = fits.ImageHDU(canvas)
    new_hdul = fits.HDUList([hdu1, hdu2])
    os.system("rm " + outname)
    new_hdul.writeto(outname, overwrite=True)
    return(outname)


def bootima(fits_list, ext, nsimul, outname, clean=True, verbose=False, mode="median", force_slice=0):
    """
    A program to calculate median of large amount of images stored in fits files.

    v2.0: First working version
    v2.1: Starting to add support for more images than RAM through chunking
    v2.2: Now using quantiles for 1sigma instead of STD for error extension.
    v2.2.1: Writing NSIMUL,NSLICES,MODE in header h0
    """


    if verbose:
        print("Bootima v2.2.1")

    outpath = os.path.dirname(outname)

    # Check if the image list fits in RAM
    it_fits = does_it_fit_in_RAM(fits_list=fits_list, ext=ext)

    if (it_fits[0]) and (force_slice==0):
        bootima_slice(fits_list=fits_list, ext=ext, nsimul=nsimul, outname=outname,
                      clean=clean, verbose=verbose, mode=mode)
    # If it doesnt fit, slice it, bootima it and then reconstruct the result
    if (not it_fits[0]) or (force_slice>0):
        print(it_fits[1])
        slicing_factor = it_fits[1]["dataset_size"]/it_fits[1]["available_memory"]
        print("- Slicing factor: " + str(slicing_factor))
        nslices=int(np.ceil(slicing_factor))

        if force_slice > 0:
            print("Forcing slicing! ")
            print("Nslices: " + str(nslices))
            nslices = force_slice

        print("- Number of slices: " + str(nslices))
        # We will always slice in NAXIS1
        slice_size = int(it_fits[1]["NAXIS1"]/nslices)
        print("- Slice size: " + str(slice_size) + " px")
        slice_archive=slice_fits(fits_list=fits_list, ext=ext, nslices=nslices)

        slice_list = []
        for i in range(nslices):
            slice_name = slice_archive[0,i].replace(".fits","_bootslice.fits")
            bootima_slice(fits_list=slice_archive[:,i], ext=1, nsimul=nsimul, outname=slice_name,
                          clean=clean, verbose=verbose, mode=mode)
            slice_list.append(slice_name)
            if clean:
                for file_to_remove in slice_archive[:,i]:
                    os.system("rm " + file_to_remove)

        # Reconstruct slices
        outname_sd = outname.replace(".fits","_sd.fits")
        reconstruct_slices(slice_list=slice_list, ext=1, outname=outname)
        reconstruct_slices(slice_list=slice_list, ext=2, outname=outname_sd)

        cmd_text = "astfits " + outname_sd + " -h1 --copy 1 --output=" + outname
        execute_cmd(cmd_text, verbose=verbose)
        execute_cmd(cmd_text = "astfits -h0 " + outname + " --update=EXTNAME,INFO", verbose=verbose)
        execute_cmd(cmd_text = "astfits -h0 " + outname + " --update=NSIMUL," + str(nsimul), verbose=verbose)
        execute_cmd(cmd_text = "astfits -h0 " + outname + " --update=MODE," + mode, verbose=verbose)
        execute_cmd(cmd_text = "astfits -h0 " + outname + " --update=NSLICES," + str(nslices), verbose=verbose)
        execute_cmd(cmd_text = "astfits -h1 " + outname + " --update=EXTNAME,MEDIAN", verbose=verbose)
        execute_cmd(cmd_text = "astfits -h2 " + outname + " --update=EXTNAME,STD", verbose=verbose)

        if (clean and os.path.exists(outname)):
            print("Cleaning temp files...")
            if os.path.exists(outname_sd):
                os.remove(outname_sd)

            for file_to_remove in slice_list:
                os.system("rm " + file_to_remove)

        if os.path.exists(outname):
            print("Simulations finished!")
            print("Final coadd: " + outname)

        else:
            print("Median output not created: Check log.")
            print("Leaving simulations for reprocessing if required.")



    return(outname)
