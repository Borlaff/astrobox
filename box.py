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
from scipy.ndimage import gaussian_filter
import bootmedian as bm
from matplotlib.path import Path


# Generate radial mask 
def create_circular_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
        
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    return(dist_from_center)

def create_angle_mask(h, w, q=1, theta=0, center=None, radius=None, pixscale=1, pitch=90):
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
        
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]

    radial_grid = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    
    x_image = np.linspace(np.min(X - center[0]),np.max(X - center[0]), w)
    y_image = np.linspace(np.min(Y - center[1]),np.max(Y - center[1]), h)
    X_image, Y_image = np.meshgrid(x_image,y_image)
    
    X_gal = (X_image*np.cos(np.radians(-theta))-Y_image*np.sin(np.radians(-theta)))/q
    Y_gal = (X_image*np.sin(np.radians(-theta))+Y_image*np.cos(np.radians(-theta)))     
    
    U_gal_pitch = X_gal*np.cos(np.radians(pitch-90)) - Y_gal*np.sin(np.radians(pitch-90))
    V_gal_pitch = X_gal*np.sin(np.radians(pitch-90)) + Y_gal*np.cos(np.radians(pitch-90))    
    
    U_ima_pitch = (U_gal_pitch*q*np.cos(np.radians(theta))-V_gal_pitch*np.sin(np.radians(theta)))
    V_ima_pitch = (U_gal_pitch*q*np.sin(np.radians(theta))+V_gal_pitch*np.cos(np.radians(theta)))      

    
    #U_ima = (U_gal_pitch*q*np.cos(np.radians(theta))-V_gal_pitch*np.sin(np.radians(theta)))
    #V_ima = (U_gal_pitch*q*np.sin(np.radians(theta))+V_gal_pitch*np.cos(np.radians(theta))) 


    angle_array = np.degrees(np.arctan2(U_ima_pitch,V_ima_pitch))
    
    angle_array[angle_array < 0] = angle_array[angle_array < 0] + 360
    
    return(angle_array)



def create_radial_mask(h, w, q=1, theta=0, center=None, radius=None, pixscale=1):
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]

    radial_grid = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    
    x_image = np.linspace(0, w-1, w) - center[0]
    y_image = np.linspace(0, h-1, h) - center[1]
    X_image, Y_image = np.meshgrid(x_image,y_image)

    X_gal = (X_image*np.cos(np.radians(-theta))-Y_image*np.sin(np.radians(-theta)))/q 
    Y_gal = (X_image*np.sin(np.radians(-theta))+Y_image*np.cos(np.radians(-theta))) 

    radial_array = np.sqrt(X_gal**2 + Y_gal**2)
    return(radial_array)


def project_image(input_image, PA, q, clean=True):
    save_fits(input_image, "projection_dummy.fits")
    
    execute_cmd("astwarp --rotate " + str(PA) + " --scale=" + str(q)+ ",1 -h0 projection_dummy.fits")
    output_image = fits.open("projection_dummy_warped.fits")[1].data
    if clean:
        os.system("rm projection_dummy.fits projection_dummy_warped.fits")
    return(output_image)

        
def save_fits(array, name):
    hdu = fits.PrimaryHDU(array)
    hdul = fits.HDUList([hdu])

    os.system("rm " + name)
    hdul.writeto(name)


def execute_cmd(cmd_text, verbose=False):
    """
    execute_cmd: This program executes in shell the input string command, printing the output for the user
    """
    #print(cmd_text)
    #with open('temp.log', 'wb') as f:
    #    process = subprocess.Popen(cmd_text, stdout=subprocess.PIPE, shell=True)
    #    for line in iter(process.stdout.readline, b''):  # With Python 3, you need iter  (process.stdout.readline, b'') (i.e. the sentinel passed to iter needs to be a binary string, since b'' != '')
    #        sys.stdout.write(line.decode(sys.stdout.encoding))
    #        f.write(line)

    #out, error = p.communicate()
    #if ((len(error) > 5) and verbose):
    #    print(error)

    #print(cmd_text)
    p = subprocess.Popen(cmd_text, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, error = p.communicate()
    try:
        output = float(out)
    except ValueError:
        output = out.decode("utf-8").replace("'" , "").replace("\n" , "")
        return(error)
    return(output)



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


def smooth_fits(fits_file, ext, sigma):
    """
    This program applies a gaussian smoothing of s=SIGMA to the array stored in the extension EXT of the FITS_FILE.
    """
    fits_image = fits.open(fits_file)
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





def interpolate_array(data):
    print("Interpolating NaNs")
    interp1 = np.array(pd.DataFrame(data).interpolate(axis=0, limit_direction="both"))
    interp1 = np.array(pd.DataFrame(interp1).interpolate(axis=1, limit_direction="both"))
    interp2 = np.array(pd.DataFrame(data).interpolate(axis=1, limit_direction="both"))
    interp2 = np.array(pd.DataFrame(interp2).interpolate(axis=0, limit_direction="both"))
    interp = (interp1 + interp2)/2.
    return(interp)


def remove_gradient(input_flc, ext, detected_name):
    input_fits = fits.open(input_flc)
    detected_fits = fits.open(detected_name)
    input_fits[ext].data = detected_fits[1].data
    os.system("rm " + input_flc)
    input_fits.verify("silentfix")
    input_fits.writeto(input_flc)
    input_fits.close()
    detected_fits.close()


def zeros_to_nan(fits_name, ext):
    print("Remove zeros from " + fits_name)
    fits_file = fits.open(fits_name, mode="update")
    fits_file[ext].data[fits_file[ext].data == 0] = np.nan
    fits_file.flush()
    fits_file.close()



def generate_ref_catalog(ref_image, ext):
    flag_name = generate_flag_ima([ref_image], ext=ext)[0]
    catalog_name = ref_image.replace(".fits",".cat")
    os.system("sex " + ref_image +
          " -c " + "/home/borlaff/GTC/SEX_files/confi.sex " +
          "-CHECKIMAGE_NAME image_segmentation.fits " +
          "-CATALOG_NAME " + catalog_name +
          " -FLAG_IMAGE " + flag_name)
    return(catalog_name)



def astalign(fits_list, outname, catalog_ref):
    fits_list_str=""
    work_path = os.path.dirname(os.path.abspath(os.path.basename(r_input_images[0])))

    for fits_input in tqdm(fits_list):

        # Emancipate HDUs
        sci1_ima = fits_input.replace(".fits", "_sci1.fits")
        os.system("astarithmetic " + fits_input + " 0 + -h1 --output=" + sci1_ima)
        sci2_ima = fits_input.replace(".fits", "_sci2.fits")
        os.system("astarithmetic " + fits_input + " 0 + -h2 --output=" + sci2_ima)

        # Generate flag images
        flag_name_1 = generate_flag_ima([sci1_ima], ext=1)[0]
        flag_name_2 = generate_flag_ima([sci2_ima], ext=1)[0]

        # Generate sextractor catalogs
        catalog_1_name = sci1_ima.replace(".fits","_cat.fits")
        os.system("sex " + sci1_ima +
                  " -c " + "/home/borlaff/GTC/SEX_files/confi.sex " +
                  " -CHECKIMAGE_NAME image_segmentation.fits " +
                  " -CATALOG_NAME " + catalog_1_name +
                  " -FLAG_IMAGE " + flag_name_1)

        catalog_2_name = sci2_ima.replace(".fits","_cat.fits")
        os.system("sex " + sci2_ima +
                  " -c " + "/home/borlaff/GTC/SEX_files/confi.sex " +
                  " -CHECKIMAGE_NAME image_segmentation.fits " +
                  " -CATALOG_NAME " + catalog_2_name +
                  " -FLAG_IMAGE " + flag_name_2)

        os.system("scamp " + catalog_1_name + " -ASTREFCAT_NAME " + catalog_ref +
                  " -c /home/borlaff/GTC/SEX_files/scamp.conf")
        os.system("scamp " + catalog_2_name + " -ASTREFCAT_NAME " + catalog_ref +
                  " -c /home/borlaff/GTC/SEX_files/scamp.conf")

        os.system("mv " + catalog_1_name.replace("_cat.fits", "_cat.head") + " " + catalog_1_name.replace("_cat.fits", ".head"))
        os.system("mv " + catalog_2_name.replace("_cat.fits", "_cat.head") + " " + catalog_2_name.replace("_cat.fits", ".head"))
        fits_list_str = fits_list_str + " " + sci1_ima + " " + sci2_ima

    os.system("swarp -c /home/borlaff/GTC/SEX_files/swarp.conf " + fits_list_str)
    outfile = work_path + "/" + outname
    os.system("mv " + work_path + "/coadd.fits" + " " + outfile)



def normalize_frame(fits_list, ext):
    """
    Normalize the data extension from the input list of files
    """
    corrected_files = np.array([])
    if isinstance(fits_list, str):
        fits_list = [fits_list]

    for fits_name in fits_list:
        #fits_file = fits.open(fits_name)
        #norma = bn.nanmedian(fits_file[ext].data)
        #if ext==1:
        #    norma = bn.nanmedian(fits_file[ext].data[850:1000,850:1000])
        #if ext==2:
        #    norma = bn.nanmedian(fits_file[ext].data[0:250,850:1000])

        #fits_file[ext].data = fits_file[ext].data/norma
        #if os.path.exists(fits_name):
        #    os.remove(fits_name)
        #fits_file.verify("silentfix")
        #fits_file.writeto(fits_name)
        #fits_file.close()
        
        print(fits_name)
        norma = execute_cmd(cmd_text = "astarithmetic " + fits_name + " -h" + str(ext) + " medianvalue -q", verbose=True)
        cmd_text = "astarithmetic " + fits_name + " " + str(norma) + " / -K -h" + str(ext)
        print(cmd_text)
        execute_cmd(cmd_text = cmd_text, verbose=True)
        if os.path.isfile(fits_name.replace(".fits", "_arith.fits")):
            os.system("rm " + fits_name)
            os.system("mv " + fits_name.replace(".fits", "_arith.fits") + " " + fits_name)        
            execute_cmd(cmd_text = "astfits -h" + str(ext)+ " " + fits_name + " --update=NORMFACT," + str(np.round(norma,8)), verbose=False)        
            execute_cmd(cmd_text = "astfits -h" + str(ext)+ " " + fits_name + " --update=NORMAL,True", verbose=False)
            corrected_files = np.append(corrected_files, fits_name)

    return(corrected_files)



def mask_fits(fits_list, ext, clean=True):
    output_list = []

    if isinstance(fits_list, str):
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



def final_skycor(input_flc, ext, nsimul=25):
    mask_1 = mask_fits(fits_list=[input_flc], ext=ext)
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
                shape_mask = mask_fits[1].data.shape
                shape_fits = fits_image[ext].data.shape
                mask_array = mask_fits[1].data
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



# A program to transform ds9 polygon vertice files to fits masks
# Alejandro Serrano Borlaff
# PhD Student - Instituto de Astrofisica de Canarias
# v.1.0 - First version (31/07/2017)
#####################################################################

# The main idea was taken from
# http://xingxinghuang.blogspot.fi/2014/05/imagebad-pixel-mask-using-ds9-region.html
# And https://stackoverflow.com/questions/3654289/scipy-create-2d-polygon-mask

def read_ds9reg(fname):
    with open(fname) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    content = content[3:]
    polygons = []
    for i in range(len(content)):
        polygon_unit = []
        polygon = content[i].replace("polygon(","").replace(")","").split(",")
        #print(len(polygon))
        #print(np.array(np.linspace(start=0,stop=len(polygon)-2,num=len(polygon)/2),dtype="int"))
        for j in np.array(np.linspace(start=0,stop=len(polygon)-2,num=int(len(polygon)/2)),dtype="int"):
            x = int(np.round(np.array(float(polygon[j]))))
            y = int(np.round(np.array(float(polygon[j+1]))))
            polygon_unit.append((x,y))
        polygons.append(polygon_unit)
    return(polygons)

def ds9tomask(fname, nx, ny, outname):
    polygons = read_ds9reg(fname)
    #nx, ny = 2000, 2000
    mask = np.zeros(shape=(ny,nx))
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x,y)).T

    for i in tqdm(range(len(polygons))):
        path = Path(polygons[i])
        grid = path.contains_points(points)
        grid = grid.reshape((ny,nx))
        mask[grid] = 1

    hdu = fits.PrimaryHDU(mask)
    os.system("rm "+outname)
    hdu.writeto(outname)
    print("File saved:"+ outname)
    print("Note: 1 is masked. 0 is not masked")
