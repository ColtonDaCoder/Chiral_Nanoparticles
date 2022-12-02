import numpy as np
#import jcmwave,time,imp,shutil,os 
from optparse import OptionParser
AOI = [i+20 for i in range(26)]
azi = [i*2 for i in range(46)]
lams = [230]

keys = {}  # Create empty dictionary for keys
for lam in lams:
    for aoi in AOI:
        parser = OptionParser()
        parser.add_option("-t", "--threads",
        action="store",type="int", dest="threads",
        help="number of threads to use")
        (options, args) = parser.parse_args()

        jcmwave.set_num_threads(options.threads)
        
        keys = {} # Create empty dictionary for keys
        results = [] # Create empty array for results
        tic = time.time() # use time() not clock() on linux system 
        # Set simulation parameters
        
        keys = {
            'AOI': ang,
            'radius': 50,
            'vacuum_wavelength': lam*1e-9,
            'uol': 1e-9,
            'display_triangulation' : 'no',
            'boundary' : 'Periodic',
            'info_level' : -1,
            'fem_degree' : 2,
            'n_refinement_steps' : 0, # Currently we get non-physical results if this is >0
            'thickness' : 50,
            'pitch' : 175, # pitch of square lattice (gammadion)
            'z_radius' : 5, # radius of curvature of dimer in z plane
            'z_radius_MSL' : 1 # maximum side length of z radius
            }


        tag_ = 'other_rotations_MM_NSL_hexagonal_lattice_radius_' + str(keys['radius']) + '_nm_structure_thickness_'  +  str(keys['thickness']) +  '_nm_pitch_' + str(keys['pitch']) + '_wavelength_' + str(keys['vacuum_wavelength']) +'nm_' + str(keys['AOI']) +'AOI'



        # material properties
        keys['n_3'] = 1.00 # index refraction of Air

            
        Au_nk = np.loadtxt('../data/Al_OMEL_mfp.nk')   #Al not GOLD!!!!!!!!!!!!!!!!!!!!!!!
        wl_Au_data = []; n_Au_real = []; n_Au_imag = []
        for data in Au_nk:
            wl_Au_data.append(data[0]*1e-9) # e-10 for [ang], e-9 for [nm], e-6 for [um]
            n_Au_real.append(data[1])
            n_Au_imag.append(data[2])

                
        # material properties
        keys['n_3'] = 1.00 # index refraction of Air

                
        azimuths = np.linspace(30, 90, 31)
        for keys['azimuth'] in azimuths:
            #print('Wavelength : %3.2f nm' % (keys['vacuum_wavelength']*1e9))
            keys['n_2'] = np.interp(keys['vacuum_wavelength'], wl_Au_data, n_Au_real) + 1j*np.interp(keys['vacuum_wavelength'], wl_Au_data, n_Au_imag)
            keys['sm_filename'] = '"'+'project_results/sm.jcm"'
            jcmwave.jcmt2jcm('./boundary_conditions.jcmt', keys)
            jcmwave.jcmt2jcm('./materials.jcmt', keys)
            jcmwave.jcmt2jcm('./project.jcmpt', keys)
            jcmwave.jcmt2jcm('./sources.jcmt', keys)
            jcmwave.jcmt2jcm('./layout.jcmt', keys)
            jcmwave.solve('./project.jcmp')
            
            ## Gather Reflected Fourier Modes (Z)
            filename_fourierModes_r = './project_results/fourier_modes_r.jcm';
            fourierModes_r = jcmwave.loadtable(filename_fourierModes_r,format='named')
            powerFlux_r = jcmwave.convert2powerflux(fourierModes_r)

            ## Reflected flux in normal direction
            P_s_t = np.sum(powerFlux_r['PowerFluxDensity'][0][:, 2]);
            P_p_t = np.sum(powerFlux_r['PowerFluxDensity'][1][:, 2]); 

            
            filename_MM = './project_results/sm.jcm'

            print(filename_MM)
            table = jcmwave.loadtable(filename_MM)
            m11 = table['Mueller_xy11'][0]
            m12 = table['Mueller_xy12'][0]
            m13 = table['Mueller_xy13'][0]
            m14 = table['Mueller_xy14'][0]
            m21 = table['Mueller_xy21'][0]
            m22 = table['Mueller_xy22'][0]
            m23 = table['Mueller_xy23'][0]
            m24 = table['Mueller_xy24'][0]
            m31 = table['Mueller_xy31'][0]
            m32 = table['Mueller_xy32'][0]
            m33 = table['Mueller_xy33'][0]
            m34 = table['Mueller_xy34'][0]
            m41 = table['Mueller_xy41'][0]
            m42 = table['Mueller_xy42'][0]
            m43 = table['Mueller_xy43'][0]
            m44 = table['Mueller_xy44'][0]

            
            

            # save data to file
            results.append([keys['vacuum_wavelength'], keys['pitch'], keys['AOI'], keys['azimuth'], P_s_t, P_p_t,m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34, m41, m42, m43, m44])
            np.savetxt('./my_results' + tag_ + '.txt', results, header='wvl[m], pitch, AOI, Azimuth, Transm_Pol-1, Transm_Pol-2, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34, m41, m42, m43, m44')
            
        toc = time.time() # use time() not clock() on linux system  
        t = toc-tic
        print ("Total runtime for "+tag_+": %6.4f s" % t)
        
