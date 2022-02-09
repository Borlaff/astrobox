#######################################################
###                                                 ###
###       IRSA SPECIFIC CLASS                       ###
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

def fullmap_irsa(wave, year, day, nside=128):
    # This IPAC table example has equatorial coordinates in decimal degrees (J2000) labeled
    # with lower-case "ra" and "dec". The observer's location "obsloc" is 0 for Earth-Sun
    # L2 and 3 for Earth (defaults to 0). "ido_view" of 1 overrides the day and computes the
    # median value for a typical spacecraft viewing zone (defaults to 1). If not present,
    # here are the defaults for the other parameters: wavelength (2.0 microns), year (2019), day (180).
    #
    # |  ra	|  dec	|  wavelength	|  year	|  day	|  obsloc	|  ido_view	|
    # |  double	|  double	|  double	|  char	|  char	|  char	|  char	|
    # |  deg	|  deg	|  microns	|  	|  	|  	|  	|
    #    10.684710	    41.268749	      1.0	    2019	    180	    0	    1
    #    92.820389	   -66.155838	    12.0	    2021	    180	    3	    0
    NSIDE = nside
    print("Approximate resolution at NSIDE {} is {:.2} deg".format(NSIDE, hp.nside2resol(NSIDE, arcmin=True) / 60))
    NPIX = hp.nside2npix(NSIDE)
    m = np.arange(NPIX)
    lon, lat = hp.pixelfunc.pix2ang(NSIDE, ipix=m, nest=False, lonlat=True)
    gc = SkyCoord(lon=lon*u.degree, lat=lat*u.degree, frame='barycentrictrueecliptic')

    db_irsa = pd.DataFrame(index = np.arange(NPIX),
                           columns=['RA', 'DEC', 'lambda', "year", "day", "obsloc", "ido_view"])

    db_irsa["RA"] = gc.icrs.ra.degree
    db_irsa["DEC"] = gc.icrs.dec.degree
    db_irsa["lambda"] = wave
    db_irsa["year"] = year
    db_irsa["day"] = day
    db_irsa["obsloc"] = 0
    db_irsa["ido_view"] = 0
    db_irsa.to_csv("full_map_noheader.csv", sep=" ", index=False, header=False)
    os.system("cat header_irsa_table.csv full_map_noheader.csv > full_map.csv")
    return(db_irsa)


def cat2hpx(lon, lat, nside, radec=True):
    """
    Convert a catalogue to a HEALPix map of number counts per resolution
    element.

    Parameters
    ----------
    lon, lat : (ndarray, ndarray)
        Coordinates of the sources in degree. If radec=True, assume input is in the icrs
        coordinate system. Otherwise assume input is glon, glat

    nside : int
        HEALPix nside of the target map

    radec : bool
        Switch between R.A./Dec and glon/glat as input coordinate system.

    Return
    ------
    hpx_map : ndarray
        HEALPix map of the catalogue number counts in Galactic coordinates

    """

    npix = hp.nside2npix(nside)

    if radec:
        eq = SkyCoord(lon, lat, 'icrs', unit='deg')
        l, b = eq.galactic.l.value, eq.galactic.b.value
    else:
        l, b = lon, lat

    # conver to theta, phi
    theta = np.radians(90. - b)
    phi = np.radians(l)

    # convert to HEALPix indices
    indices = hp.ang2pix(nside, theta, phi)
    idx, counts = np.unique(indices, return_counts=True)

    # fill the fullsky map
    hpx_map = np.zeros(npix, dtype=int)
    hpx_map[idx] = counts

    return hpx_map




def read_irsa_map(filename):
    # pd.read_table("/home/borlaff/ESA/Euclid/SKY/background.tbl.txt", skiprows=24, sep=" ")
    db_irsa_skybg = np.loadtxt(filename, skiprows=23)
    db_irsa_skybg = pd.DataFrame(db_irsa_skybg)
    # |objname|ra|dec|zody|ism|stars| cib |totbg|wavelength  |year    |day     |selon   |obsloc  |obsver  |ido_view|errmsg
    db_irsa_skybg.columns = ['ra', 'dec', 'zody', 'ism', 'stars', 'cib', 'totbg', 'wave', 'year', 'day', 'selon', 'obsloc', 'obsver', 'ido_view']
    return(db_irsa_skybg)

def read_irsa_wave_cat(filelist):
# pd.read_table("/home/borlaff/ESA/Euclid/SKY/background.tbl.txt", skiprows=24, sep=" ")

    db_sky_05 = read_irsa_map(filelist[0])
    db_sky_06 = read_irsa_map(filelist[1])
    db_sky_07 = read_irsa_map(filelist[2])
    db_sky_08 = read_irsa_map(filelist[3])
    db_sky_09 = read_irsa_map(filelist[4])
    db_sky_10 = read_irsa_map(filelist[5])

    # Check ncols
    if not len(db_sky_05.iloc[:,0]) == len(db_sky_06.iloc[:,0]) == len(db_sky_07.iloc[:,0]) == len(db_sky_08.iloc[:,0]) == len(db_sky_09.iloc[:,0]) == len(db_sky_10.iloc[:,0]):
        print("Error: Background files do not have the same number of rows")
        return(1)
    else:
        print("Number of columns OK")
        nrows = len(db_sky_05.iloc[:,0])

# Check wavelengths
    input_wave = np.array([db_sky_05["wave"][0], db_sky_06["wave"][0], db_sky_07["wave"][0],
                           db_sky_08["wave"][0],db_sky_09["wave"][0],db_sky_10["wave"][0]])
    if (input_wave[0] == 0.5) & (input_wave[1] == 0.6) & (input_wave[2] == 0.7) & (input_wave[3] == 0.8) & (input_wave[4] == 0.9) & (input_wave[5] == 1.0):
        print("Wavelength crosscheck: OK")
    else:
        print("Error: Lambda dont match file names. Check files")
        return(1)

    # Check dates
    DRA =  db_sky_05.iloc[:,0] + db_sky_06.iloc[:,0] + db_sky_07.iloc[:,0] - db_sky_08.iloc[:,0] - db_sky_09.iloc[:,0] - db_sky_10.iloc[:,0]
    DDEC = db_sky_05.iloc[:,1] + db_sky_06.iloc[:,1] + db_sky_07.iloc[:,1] - db_sky_08.iloc[:,1] - db_sky_09.iloc[:,1] - db_sky_10.iloc[:,1]
    if np.abs(np.sum(DRA + DDEC)) < 0.00001:
        print("RA DEC crosscheck: OK")
    else:
        print("Error: RA DEC coordinates do not match")
        print(np.sum(DRA + DDEC))
        print(DRA)
        print(DDEC)
        print(db_sky_05.iloc[:,0])
        print(db_sky_06.iloc[:,0])

    zody = []
    ism = []
    stars = []
    cib = []
    totbg = []

    for i in range(nrows):
        zody_single_pointing = np.array([db_sky_05["zody"][i], db_sky_06["zody"][i], db_sky_07["zody"][i],
                                         db_sky_08["zody"][i], db_sky_09["zody"][i], db_sky_10["zody"][i]])

        ism_single_pointing = np.array([db_sky_05["ism"][i], db_sky_06["ism"][i], db_sky_07["ism"][i],
                                        db_sky_08["ism"][i], db_sky_09["ism"][i], db_sky_10["ism"][i]])

        stars_single_pointing = np.array([db_sky_05["stars"][i], db_sky_06["stars"][i], db_sky_07["stars"][i],
                                        db_sky_08["stars"][i], db_sky_09["stars"][i], db_sky_10["stars"][i]])

        cib_single_pointing = np.array([db_sky_05["cib"][i], db_sky_06["cib"][i], db_sky_07["cib"][i],
                                        db_sky_08["cib"][i], db_sky_09["cib"][i], db_sky_10["cib"][i]])

        totbg_single_pointing = zody_single_pointing + ism_single_pointing + stars_single_pointing + cib_single_pointing

        zody.append(zody_single_pointing)
        ism.append(ism_single_pointing)
        stars.append(stars_single_pointing)
        cib.append(cib_single_pointing)
        totbg.append(totbg_single_pointing)



# Check if the objects are the same for the different filters
    db_irsa = {"ra": np.array(db_sky_05["ra"]),
               "dec": np.array(db_sky_05["dec"]),
               "lambda": input_wave,
               "year":   np.array(db_sky_05["year"]),
               "day":    np.array(db_sky_05["day"]),
               "obsloc": np.array(db_sky_05["obsloc"]),
               "ido_view": np.array(db_sky_05["ido_view"]),
               "zody": zody,
               "ism": ism,
               "stars": stars,
               "cib": cib,
               "totbg": totbg
              }
    return(db_irsa)


def launch_irsa_query(ra, dec, wavelength, year, day):
    ra_str = str(round(ra,7)).zfill(7)
    dec_str = str(round(dec,7)).zfill(7)
    wave_str = str(round(wavelength,2)).zfill(2)
    year_str = str(year)
    day_str = str(day).zfill(2)

    outfile="irsa_ra"+ra_str+"_dec"+dec_str+"_wave"+wave_str+"_y"+year_str+"_d"+day_str+".xml"

    if os.stat(outfile).st_size > 900:
        return(outfile)


    website="https://irsa.ipac.caltech.edu/cgi-bin/BackgroundModel/nph-bgmodel?"
    cmd = 'wget -O ' + outfile + ' "' + website + 'locstr='+str(ra)+'+'+str(dec)+'+equ+j2000&wavelength='+str(wavelength)+'&year='+str(year)+'&day='+str(day)+'&obslocin=0&ido_viewin=0"'
    print(cmd)
    os.system("rm " + outfile)
    os.mknod(outfile)
    execute_cmd(cmd, verbose=True)
    while os.stat(outfile).st_size <= 16:
        print("Launching...")
        execute_cmd(cmd, verbose=True)
    return(outfile)


def read_irsa_query(infile):
    #print("Reading " + infile)
    lstValue = []
    i = 0

    data=infile
    year = [i for i in xmliter(data, "year")][0]
    day = [i for i in xmliter(data, "day")][0]


    for d in tqdm(xmliter(data, "statistics")):
        d['zody'] = float(d['zody'].replace("(MJy/sr)", ""))
        d['cib'] = float(d['cib'].replace("(MJy/sr)", ""))
        d['stars'] = float(d['stars'].replace("(MJy/sr)", ""))
        d['ism'] = float(d['ism'].replace("(MJy/sr)", ""))
        d['totbg'] = float(d['totbg'].replace("(MJy/sr)", ""))

        d['ra'] = float(d['refCoordinate'].split(" ")[0])
        d['dec'] = float(d['refCoordinate'].split(" ")[1])
        d.pop("refCoordinate")
        d['year'] = year
        d['day'] = day
        lstValue.append(d)
        i = i + 1

    return(lstValue)



def download_query_euclid_mission_plan(data="/lhome/aserrano/PLAN/SPV2.1_cor.xml"):
    obsplan = tars.read_euclid_mission_plan(data)
    wave = [0.5,0.6,0.7,0.8,0.9,1.0]
    launch_query = True # CAREFUL : This takes at least 16 hours
    if launch_query:
        for i in wave:
            npointings = len(obsplan["RA"])
            arguments = zip(np.array(obsplan["RA"]), np.array(obsplan["DEC"]),
                            np.array([i]*npointings), np.array(obsplan["Year"]), np.array(obsplan["Day_year"]))

            pool = multiprocessing.Pool(processes=100)
            test = pool.starmap(launch_irsa_query, arguments)
            pool.terminate()
