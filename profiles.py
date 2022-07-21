import sys
import os
import glob
from tqdm import tqdm
import bootmedian
import pandas as pd
import numpy as np



def create_radial_mask(xsize, ysize, q=1, theta=0, center=None, radius=None):
    if center is None: # use the middle of the image
        center = (int(xsize/2), int(ysize/2))

    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], xsize-center[0], ysize-center[1])
    X, Y = np.ogrid[:xsize, :ysize]

    radial_grid = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    x_image = np.linspace(np.min(X - center[0]),np.max(X - center[0]), xsize)
    y_image = np.linspace(np.min(Y - center[1]),np.max(Y - center[1]), ysize)

    X_image, Y_image = np.meshgrid(x_image,y_image) # Images with the observer X and Y coordinates

    X_gal = (X_image*np.cos(np.radians(-theta))-Y_image*np.sin(np.radians(-theta)))/q # X in galactic frame
    Y_gal = (X_image*np.sin(np.radians(-theta))+Y_image*np.cos(np.radians(-theta)))   # Y in galactic frame

    radial_array = np.sqrt(X_gal**2 + Y_gal**2)
    return(radial_array)


def create_circular_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    return(dist_from_center)


def make_profile(image, radial_mask, rbins, nsimul, mode="median"):

    profile = np.zeros((len(rbins)-1,7))
    rad = np.zeros((len(rbins)-1,3))

    model_image = np.zeros(radial_mask.shape)
    model_image_s1up = np.zeros(radial_mask.shape)
    model_image_s1down = np.zeros(radial_mask.shape)

    model_image[:,:] = np.nan
    model_image_s1up[:,:] = np.nan
    model_image_s1down[:,:] = np.nan


    for i in tqdm(range(len(rbins)-1)):
        pixels_to_bin = np.where((radial_mask >= rbins[i]) & (radial_mask < rbins[i+1]))
        rad[i,0] = np.median(radial_mask[pixels_to_bin])
        rad[i,1] = np.max(radial_mask[pixels_to_bin])
        rad[i,2] = np.min(radial_mask[pixels_to_bin])
        boot_g = bootmedian.bootmedian(image[pixels_to_bin], nsimul=nsimul, mode=mode)
        profile[i,0] = boot_g["median"]
        profile[i,1] = boot_g["s1_up"]
        profile[i,2] = boot_g["s1_down"]
        profile[i,3] = boot_g["s2_up"]
        profile[i,4] = boot_g["s2_down"]
        profile[i,5] = boot_g["s3_up"]
        profile[i,6] = boot_g["s3_down"]

        model_image[pixels_to_bin] = boot_g["median"]
        model_image_s1up[pixels_to_bin] = boot_g["s1_up"]
        model_image_s1down[pixels_to_bin] = boot_g["s2_down"]


    df = pd.DataFrame(data=np.array([rad[:,0],
                                     rad[:,1],
                                     rad[:,2],
                                     profile[:,0],
                                     profile[:,1],
                                     profile[:,2],
                                     profile[:,3],
                                     profile[:,4],
                                     profile[:,5],
                                     profile[:,6]]).T,
                      columns=["r", "r_s1up", "r_s1down", "int", "int_s1up", "int_s1down", "int_s2up", "int_s2down", "int_s3up", "int_s3down"])


    return([df, model_image, model_image_s1up, model_image_s1down])
