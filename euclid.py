
#######################################################
###                                                 ###
###     EUCLID SPECIFIC CLASS                       ###
###                                                 ###
#######################################################


class euclid:
    def euclid_normalize_mask(fits_list):
        """
        euclid.normalize_mask  - Antiguo normalize_eur_mask
        """
        if not isinstance(fits_list, list):
            fits_list = [fits_list]

        corrected_files = np.array([])
        for fits_name in fits_list:
            fits_file = fits.open(fits_name)
            for i in np.linspace(1,36,36).astype("int"):
                # Translate to GNUASTRO
                # Normalize as a function of width and center
                print("Normalizing CCD " + str(i))
                # execute_cmd(cmd_text="astarithmetic -h " + str(j) + " " + outname + " -h" + str(j+2) + " " + outname + " 0 gt nan where")
                fits_file[i].data = np.divide(fits_file[i].data, bn.nanmedian(fits_file[i].data[1798:2298, 1818:2318]))

            if os.path.exists(fits_name):
                os.remove(fits_name)
            fits_file.verify("silentfix")
            fits_file.writeto(fits_name)
            fits_file.close()
            execute_cmd(cmd_text = "astfits -h0 " + fits_name + " --update=NORMAL,True")
            corrected_files = np.append(corrected_files, fits_name)
        return(corrected_files)


    def mask_euc_exposure(fits_list):
        output_list = []
        if not isinstance(fits_list, list):
            fits_list=[fits_list]

        vis_sci_extensions = np.linspace(1, 106, 36).astype("int")
        for i in tqdm(fits_list):
            file_name = os.path.abspath(i)
            outname = file_name.replace(".fits", "_mask.fits")
            if os.path.isfile(outname):
                os.remove(outname)
            #execute_cmd(cmd_text="cp " + file_name + " " + outname)
        # Ensure everything is clean
            # Cleaning everything for the next loop
            os.system("rm dummy.fits")
            os.system("rm dq.fits")
            os.system("rm dq_mask.fits")
            os.system("rm masked.fits")
            os.system("rm detections_mask.fits")
            os.system("rm dummy_cr_masked.fits")
            os.system("rm detections.fits")

            for j in (vis_sci_extensions):
                # ##################################################################################
                # Hay que interpolar los CR antes de enmascarar, o perdemos demasiado area #########
                ####################################################################################
                cmd_text = "astfits -h" + str(j) + " " + file_name + " --copy=" + str(j) + " --output=dummy.fits"
                execute_cmd(cmd_text, verbose=True)
                cmd_text = "astfits -h" + str(j+2) + " " + file_name + " --copy=" + str(j+2) + " --output=dq.fits"
                execute_cmd(cmd_text, verbose=True)
                cmd_text = "astarithmetic -h1 dummy.fits -h1 dq.fits 8 eq nan where --output=dummy_cr_masked.fits"
                execute_cmd(cmd_text, verbose=True)

                cmd_text = "astnoisechisel --keepinputdir --tilesize=50,50 --interpnumngb=3 dummy_cr_masked.fits -h1 --output=detections.fits"
                execute_cmd(cmd_text, verbose=True)
                # Translate to GNUASTRO
                # Mask to 1 the j + 2 extension when is detected and dq extension is 0
                cmd_text = "astarithmetic --keepinputdir -h2 detections.fits 0 gt --output=detections_mask.fits"
                execute_cmd(cmd_text, verbose=True)
                cmd_text = "astarithmetic --keepinputdir -h" + str(j+2) + " " + file_name + " 0 gt --output=dq_mask.fits"
                execute_cmd(cmd_text, verbose=True)
                cmd_text = "astarithmetic --keepinputdir -h1 detections_mask.fits -h1 dq_mask.fits or --output=mask.fits"
                execute_cmd(cmd_text, verbose=True)
                cmd_text = "astarithmetic --keepinputdir -h" + str(j) + " " + file_name + " -h1 mask.fits nan where --output=masked.fits"
                execute_cmd(cmd_text, verbose=True)

                cmd_text = "astfits -h1 masked.fits --copy=1 --output=" + outname
                execute_cmd(cmd_text, verbose=True)

                # Cleaning everything for the next loop
                os.system("rm dummy.fits")
                os.system("rm dq.fits")
                os.system("rm dq_mask.fits")
                os.system("rm masked.fits")
                os.system("rm detections_mask.fits")
                os.system("rm dummy_cr_masked.fits")
                os.system("rm detections.fits")
            output_list.append(outname)
        return(output_list)



    def dq_mask_euc_exposure(fits_list):
        output_list = []
        if not isinstance(fits_list, list):
            fits_list=[fits_list]

        vis_sci_extensions = np.linspace(1, 106, 36).astype("int")
        for i in tqdm(fits_list):
            file_name = os.path.abspath(i)
            outname = file_name.replace(".fits", "_mask.fits")
            if os.path.isfile(outname):
                os.remove(outname)
            #execute_cmd(cmd_text="cp " + file_name + " " + outname)
        # Ensure everything is clean
            # Cleaning everything for the next loop
            os.system("rm dummy.fits")
            os.system("rm dq.fits")
            os.system("rm dq_mask.fits")
            os.system("rm masked.fits")
            os.system("rm detections_mask.fits")
            os.system("rm dummy_cr_masked.fits")
            os.system("rm detections.fits")

            for j in (vis_sci_extensions):
                # ##################################################################################
                # Hay que interpolar los CR antes de enmascarar, o perdemos demasiado area #########
                ####################################################################################
                cmd_text = "astfits -h" + str(j) + " " + file_name + " --copy=" + str(j) + " --output=dummy.fits"
                execute_cmd(cmd_text, verbose=True)
                # Mask to 1 the j + 2 extension when is detected and dq extension is 0
                cmd_text = "astarithmetic --keepinputdir -h" + str(j+2) + " " + file_name + " 0 gt --output=dq_mask.fits"
                execute_cmd(cmd_text, verbose=True)
                cmd_text = "astarithmetic --keepinputdir -h" + str(j) + " " + file_name + " -h1 dq_mask.fits nan where --output=masked.fits"
                execute_cmd(cmd_text, verbose=True)
                cmd_text = "astfits -h" + str(j) + " masked.fits --copy=1 --output=" + outname
                execute_cmd(cmd_text, verbose=True)
                # Now we create the weight map, as the 1/sqrt(RMS) and setting dq != 0 to 0
                # We create the 1/sqrt(RMS) array
                cmd_text = "astarithmetic --keepinputdir -h" + str(j+1) + " " + file_name + " -0.5f pow --output=inv_sqrt_rms.fits"
                execute_cmd(cmd_text, verbose=True)
                # We mask the 1/sqrt(RMS) array with dq_mask.fits
                cmd_text = "astarithmetic --keepinputdir -h1 inv_sqrt_rms.fits -h1 dq_mask.fits 0 where --output=weight.fits"
                execute_cmd(cmd_text, verbose=True)

                cmd_text = "astfits -h" + str(j+1) + " weight.fits --copy=1 --output=" + outname
                execute_cmd(cmd_text, verbose=True)
                cmd_text = "astfits -h" + str(j+2) + " " + file_name + " --copy=" + str(j+2) + " --output=" + outname
                execute_cmd(cmd_text, verbose=True)

                # Cleaning everything for the next loop
                os.system("rm dummy.fits")
                os.system("rm dq_mask.fits")
                os.system("rm masked.fits")

            output_list.append(outname)
        return(output_list)


    def create_skyflat_local(nexp):
        vis_sci_extensions = np.linspace(1, 106, 36).astype("int")
        exp_list = glob.glob("./EUC_VIS_SWL-DET-*00.00.fits")
        #masked_list = mask_euc_exposure(exp_list)
        #normalize_eur_mask(fits_list=masked_list)
        masked_list = glob.glob("EUC*mask.fits")
        fits_list = []
        ext_list_single = np.linspace(1,36,36).astype("int")
        ext_list = np.array([])

        for i in range(len(masked_list)):
            fits_list = fits_list + [masked_list[i]]*36
            ext_list = np.concatenate((ext_list, ext_list_single), axis=None)
        ext_list = ext_list.astype("int")
        fits_list = np.array(fits_list)

        indexes = np.random.randint(0, len(ext_list), nexp).astype("int")

        print(fits_list[indexes])
        print(ext_list[indexes])
        print(nexp)

        nsimul = np.max(np.array([int(nexp*np.log10(nexp)),10]))
        print(nsimul)
        bootima(fits_list=fits_list[indexes], ext=ext_list[indexes], nsimul=nsimul, outname="coadd_nexp" + str(nexp).zfill(4) + ".fits", clean=True, verbose=True)




    def read_euclid_mission_plan(data):
        pointValue = []
        obsValue = []
        i = 0

        print("Reading pointing requests")
        for d in tqdm(xmliter(data, "ObservationRequest")):

            if not isinstance(d["PointingRequest"], list):
                d["PointingRequest"] = [d["PointingRequest"]]

            for i in d["PointingRequest"]:
                #print(i)
                i["ObservationType"] = d["ObservationType"]
                i["MissionPhase"] = d["MissionPhase"]
                i["SurveyId"] = d["SurveyId"]

            pointValue = pointValue + d["PointingRequest"]
        db = pd.DataFrame(pointValue)


        nrows = len(db.iloc[:,0])
        db_small = pd.DataFrame(index = np.arange(nrows), columns=['ID', 'MissionPhase', 'ObservationType',
                                                                   'SurveyId', 'MJD2000', 'StartTime', 'Year', 'Day_year',
                                                                   'Lon', 'Lat','RA', 'DEC', 'PA', "exptime"])

        db_small["MissionPhase"] = db.iloc[:,]["MissionPhase"]
        db_small["ObservationType"] = db.iloc[:,]["ObservationType"]
        db_small["SurveyId"] = db.iloc[:,]["SurveyId"]

        db_small["Lon"] = np.array([float(i["Longitude"]) for i in db.iloc[:,]["Attitude"]])
        db_small["Lat"] = np.array([float(i["Latitude"]) for i in db.iloc[:,]["Attitude"]])
        db_small["PA"] = np.array([float(i["PositionAngle"]) for i in db.iloc[:,]["Attitude"]])

        print("Transforming coordinates...")
        gc = SkyCoord(lon=db_small["Lon"]*u.degree, lat=db_small["Lat"]*u.degree, frame='barycentrictrueecliptic')

        db_small["ID"] = db.iloc[:,0]
        db_small["MJD2000"] = np.array(db.iloc[:,]["Mjd2000"]).astype("float")
        db_small["StartTime"] = db.iloc[:,]["StartTime"]

    #    t = [Time(db_small["StartTime"], format='isot', scale='utc') for
        print("Time dates reshaping...")
        t = [Time(i, format='isot', scale='utc') for i in db_small["StartTime"]]
        db_small["Year"] = np.array([i.datetime.year for i in t])
    #    tt = t.datetime.timetuple() # We use tm_yday transforming t to tt (tuple time)
        tt = [i.datetime.timetuple() for i in t]
        db_small["Day_year"] = np.array([i.tm_yday for i in tt])
        db_small["RA"] = gc.icrs.ra.degree
        db_small["DEC"] = gc.icrs.dec.degree
        db_small["exptime"] = np.array(db.iloc[:,]["Duration"]).astype("float")
        print("End of line")
        return(db_small)



    def euclid_mission_plan_to_irsa(db_emp, limit=0):

        nrows = len(db_emp.iloc[:,0])
        ncols = len(db_emp.iloc[0,:])


        db_irsa = pd.DataFrame(index = np.arange(nrows),
                               columns=['RA', 'DEC', 'lambda', "year", "day", "obsloc", "ido_view"])

        if(limit != 0):
            index = np.random.randint(low=0, high=nrows, size=int(limit))
            db_irsa = db_irsa.iloc[index,:]

        wavelengths = np.linspace(5000,10000,6)


        for i in wavelengths:
            db_irsa["RA"] = np.array(db_emp["RA"])
            db_irsa["DEC"] = np.array(db_emp["DEC"])
            db_irsa["lambda"] = i/10000.
            db_irsa["year"] = np.array(db_emp["Year"])
            db_irsa["day"] = np.array(db_emp["Day_year"])
            db_irsa["obsloc"] = 0
            db_irsa["ido_view"] = 0
            wave_filename_noheader = "IRSA_query_lambda" + str(int(i)) + "_noheader.csv"
            wave_filename_header = "IRSA_query_lambda" + str(int(i)) + ".csv"
            db_irsa.to_csv(wave_filename_noheader, sep=" ", index=False, header=False)
            os.system("split " + wave_filename_noheader + " IRSA_query_"+str(int(i))+"_ -l 20000 -d ")
            for i in glob.glob("IRSA_query_"+str(int(i))+"*"):
                os.system("cat header_irsa_table.csv " + i + " > " + i + ".csv")
                os.remove(i)
            os.remove(wave_filename_noheader)

        return(db_irsa)



    def emancipate_exposure(fits_list, mode='full', mask=False, dqlim=0):
        # This program operates over a list of Euclid VIS exposures. If the input is an image name (string), it puts it
        # into a single element list
        output_list=[]
        if not isinstance(fits_list, list):
            fits_list=[fits_list]

        # The structure of VIS exposures is:
        # Total fits extensions = 108
        # 1st - empty Primary HDU
        # n = [0, 36] - 36 is the number of CCDs of VIS
        # 3n + 1 - Sci image
        # 3n + 2 - RMS frame
        # 3n + 3 - Mask frame

        vis_sci_extensions = np.linspace(1,106,36).astype('int')
        vis_rms_extensions = np.linspace(1,106,36).astype('int') + 1
        vis_msk_extensions = np.linspace(1,106,36).astype('int') + 2

        for i in fits_list:
            output_exposure = []
            fits_file = fits.open(i)
            nextensions = len(fits_file)
            #print(nextensions)
            print(i)

            for sci_ext, rms_ext, mks_ext in tqdm(zip(vis_sci_extensions, vis_rms_extensions, vis_msk_extensions), total=len(vis_sci_extensions)):

                outname = i.replace('.fits','_ext' + str(sci_ext).zfill(3) + '.fits')
                # We create a 3 ext fits file per CCD - Usual input for most programs (GnuAstro, Montage)
                new_hdul = fits.HDUList()
                new_hdul.append(fits.PrimaryHDU())
                new_hdul.append(fits.ImageHDU())
                new_hdul[0].header = fits_file[0].header
                new_hdul[0].header["EXT_ORIG"] = 0
                new_hdul[0].header["NAMEORIG"] = os.path.basename(i)

                if mode=='full':
                    new_hdul.append(fits.ImageHDU())
                    new_hdul.append(fits.ImageHDU())

                new_hdul[1].header = fits_file[sci_ext].header
                new_hdul[1].header["EXT_ORIG"] = sci_ext
                new_hdul[1].data = fits_file[sci_ext].data
                if mask:
                    new_hdul[1].data[np.where(fits_file[mks_ext].data>dqlim)]=np.nan

                if mode=='full':
                    new_hdul[2].header = fits_file[rms_ext].header
                    new_hdul[2].data = fits_file[rms_ext].data
                    new_hdul[2].header["EXT_ORIG"] = rms_ext
                    new_hdul[3].header = fits_file[mks_ext].header
                    new_hdul[3].data = fits_file[mks_ext].data
                    new_hdul[3].header["EXT_ORIG"] = mks_ext

                if os.path.exists(outname):
                    os.remove(outname)
                new_hdul.verify('silentfix')
                new_hdul.writeto(outname)
                output_exposure.append(outname)
            output_list.append(output_exposure)
            if len(output_list) == 1:
                output_list = output_list[0]
        return(output_list)



    def join_exposure(fits_list, output_dir):
        # This program operates over a list of Euclid VIS emancipated CCD exposures. If the input is an image name (string), it puts it
        # into a single element list
        output_list=[]
        if not isinstance(fits_list, list):
            fits_list=[fits_list]

        # First we create the empty fits list with 109 extensions
        new_hdul = fits.HDUList()
        new_hdul.append(fits.PrimaryHDU())

        vis_extensions = np.linspace(1,108,108).astype('int')
        for j in vis_extensions:
            new_hdul.append(fits.ImageHDU())

        fits_file = fits.open(fits_list[0])
        outname_0 = fits_file[0].header["NAMEORIG"]

        for fits_name in fits_list:
            fits_file = fits.open(fits_name)
            if fits_file[0].header["NAMEORIG"] != outname_0:
                return("ERROR! Individual input files from different exposures")
            # We create a 3 ext fits file per CCD - Usual input for most programs (GnuAstro, Montage)
            new_hdul[0].header = fits_file[0].header
            new_hdul[0].data   = fits_file[0].data
            new_hdul[fits_file[1].header["EXT_ORIG"]].header = fits_file[1].header
            new_hdul[fits_file[1].header["EXT_ORIG"]].data   = fits_file[1].data
            new_hdul[fits_file[2].header["EXT_ORIG"]].header = fits_file[2].header
            new_hdul[fits_file[2].header["EXT_ORIG"]].data   = fits_file[2].data
            new_hdul[fits_file[3].header["EXT_ORIG"]].header = fits_file[3].header
            new_hdul[fits_file[3].header["EXT_ORIG"]].data   = fits_file[3].data

        outname_0 = output_dir + outname_0
        if os.path.exists(outname_0):
            os.remove(outname_0)
        new_hdul.verify('silentfix')
        new_hdul.writeto(outname_0)
        output_list.append(outname_0)

        return(output_list)




    def inject_model_into_precal_single(precal_name, model_name):
        # Lets reproject the simulation file to the exposure CCDs planes
        target_projection = fits.open(precal_name, mode="update")
        simulated_frame = fits.open(model_name)
        array, footprint = reproject_interp(simulated_frame[0], target_projection[1].header)
        array[np.isnan(array)] = 0
        array = array/target_projection[1].header['FLXSCALE']
        target_projection[1].data = target_projection[1].data + array
        target_projection.flush()
        target_projection.close()
        return(precal_name)


    def inject_model_into_precal(precal_list, model_name):
        # This program operates over a list of Euclid VIS exposures. If the input is an image name (string), it puts it
        # into a single element list
        print("Injecting " + model_name)
        output_list=[]
        if not isinstance(precal_list, list):
            precal_list=[precal_list]
        arguments = zip(np.array(precal_list), np.array([model_name]*len(precal_list)))

        nproc = multiprocessing.cpu_count() - 2

        pool = multiprocessing.Pool(processes=nproc)

        for _ in tqdm(pool.starmap(inject_model_into_precal_single, arguments), total=len(precal_list)):
            pass
        pool.terminate()
        return(precal_list)


    def inject_model_euclid_exp(fits_list, model_list, output_dir, clean=True):

        output_list=[]
        if not isinstance(fits_list, list):
            fits_list=[fits_list]
        if not isinstance(model_list, list):
            model_list=[model_list]

        for i in fits_list:
            separated_pointing_list = []
            separated_exposures = emancipate_exposure(fits_list = i, mode='full', mask=False, dqlim=0)
            separated_pointing_list.append(separated_exposures)
            for j in model_list:
                inject_model_into_precal(separated_exposures, j)
            injected_exposure = join_exposure(separated_exposures, output_dir)
            if clean:
                print("Cleaning temp files")
                for j in separated_exposures:
                    if os.path.exists(j):
                        os.remove(j)
        print("All models injected in " + output_dir + ": Done")




    def create_model_ima(model_dir, output_name, z, mu0, FOV, RA0, DEC0):
        # We read only a subset of particle properties (positions, to save memory)
        part = gizmo_read.read.Read.read_snapshot(species=('star','gas'), properties=['position'], directory=model_dir)

        # And save them to the stars and gas pandas dataframes
        # For indexing in pandas : stars.loc[x,y]
        stars=pd.DataFrame(np.array(part["star"]["position"]))
        gas=pd.DataFrame(np.array(part["gas"]["position"]))

        # We calculate the transformations for kpc to pix as a function of z (Distance)
        kpc_arcsec = cosmo.kpc_proper_per_arcmin(z)/60.
        arcsec_pix = 0.1 # Euclid VIS pixscale

        axis_obs = np.array([0,1])
        # Transform from kpc to pix
        print('Transforming coordinates to pixel space')
        stars.loc[:,axis_obs[0]] = stars.loc[:,axis_obs[0]]/(kpc_arcsec*arcsec_pix)
        stars.loc[:,axis_obs[1]] = stars.loc[:,axis_obs[1]]/(kpc_arcsec*arcsec_pix)
        gas.loc[:,axis_obs[0]] = gas.loc[:,axis_obs[0]]/(kpc_arcsec*arcsec_pix)
        gas.loc[:,axis_obs[1]] = gas.loc[:,axis_obs[1]]/(kpc_arcsec*arcsec_pix)

        # We create the image
        print('Creating stellar image')
        image_stars = np.histogram2d(x=stars.loc[:,axis_obs[0]], y=stars.loc[:,axis_obs[1]], bins=FOV, range=[[-FOV/2,FOV/2],[-FOV/2,FOV/2]], normed=None, weights=None, density=None)
        print('Creating gas image')
        image_gas =   np.histogram2d(x=gas.loc[:,axis_obs[0]],   y=gas.loc[:,axis_obs[1]], bins=FOV, range=[[-FOV/2,FOV/2],[-FOV/2,FOV/2]], normed=None, weights=None, density=None)
        data = image_stars[0] + image_gas[0]


        # Convolve by the Euclid PSF
        psf = fits.open('/localdata/Borlaff/EMDB/kernel.fits')
        print('Convolving image with LARGE PSF')
        data_low = convolve_fft(data, psf[1].data,  allow_huge=True)
        psf = fits.open('/localdata/Borlaff/EMDB/psf_VIS_centred.fits')
        print('Convolving image with Euclid VIS PSF')
        data[np.where(data ==1)] = 0
        data_high = convolve_fft(data, psf[1].data,  allow_huge=True)
        data = (data_low + data_high)/2.

        # Photometry
        # What is the mean particle density on the central pixels?
        print('Calibrating photometry')
        skybg = bn.median(data[0:int(FOV/10), 0:int(FOV/10)])
        data = data - skybg
        central_density = bn.median(data[int(FOV/2-5):int(FOV/2+5), int(FOV/2-5):int(FOV/2+5)])
        int0 = (arcsec_pix**2)*10**((24.445 - mu0)/2.5) # Central intensity for the mu0 set by the user
        photometry_correction = int0/central_density
        data = data*photometry_correction

        # We add a fake centred WCS and save the fits file
        hdu = fits.PrimaryHDU(data=data)

        print('Saving fake WCS')
        hdu.header['WCSAXES'] = 2
        hdu.header['CRPIX1'] = FOV/2.+0.5
        hdu.header['CRPIX2'] = FOV/2.+0.5
        hdu.header['CRVAL1'] =  RA0
        hdu.header['CRVAL2'] =  DEC0
        hdu.header['CTYPE1'] = 'RA---TAN'
        hdu.header['CTYPE2'] = 'DEC--TAN'
        hdu.header['RA'] = RA0
        hdu.header['DEC'] = DEC0
        hdu.header['CD1_1'] = 2.521185192875E-05
        hdu.header['CD1_2'] = 1.173845066278E-05
        hdu.header['CD2_1'] = 1.162545338166E-05
        hdu.header['CD2_2'] = -2.537923352533E-05

        print('Saving file: ' + output_name)
        if os.path.exists(output_name):
            os.remove(output_name)
        hdu.verify("silentfix")
        hdu.writeto(output_name)
