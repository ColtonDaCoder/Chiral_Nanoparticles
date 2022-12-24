import numpy as np
import json
import jcmwave,time,imp,shutil,os 
#from optparse import OptionParser
AOI = [i+20 for i in range(26)]
AZI = [i*2 for i in range(46)]
WVL = 210

keys = {}  # Create empty dictionary for keys
dict = {}
for aoi in AOI:
    #parser = OptionParser()
    #parser.add_option("-t", "--threads",
    #action="store",type="int", dest="threads",
    #help="number of threads to use")
    #(options, args) = parser.parse_args()

    #jcmwave.set_num_threads(options.threads)
    
    keys = {} # Create empty dictionary for keys
    dict[aoi] = {}
    results = [] # Create empty array for results
    tic = time.time() # use time() not clock() on linux system 
    # Set simulation parameters
    
    keys = {
        'AOI': aoi,
        'radius': 50,
        'vacuum_wavelength': WVL*1e-9,
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

        
    Au_nk = np.loadtxt('Al_OMEL_mfp.nk')   #Al not GOLD!!!!!!!!!!!!!!!!!!!!!!!
    wl_Au_data = []; n_Au_real = []; n_Au_imag = []
    for data in Au_nk:
        wl_Au_data.append(data[0]*1e-9) # e-10 for [ang], e-9 for [nm], e-6 for [um]
        n_Au_real.append(data[1])
        n_Au_imag.append(data[2])

            
    # material properties
    keys['n_3'] = 1.00 # index refraction of Air

            
    for azi in AZI:
        #print('Wavelength : %3.2f nm' % (keys['vacuum_wavelength']*1e9))
        keys['n_2'] = np.interp(keys['vacuum_wavelength'], wl_Au_data, n_Au_real) + 1j*np.interp(keys['vacuum_wavelength'], wl_Au_data, n_Au_imag)
        keys['sm_filename'] = '"'+'project_results/sm.jcm"'
        keys['azimuth'] = azi
        jcmwave.jcmt2jcm('./boundary_conditions.jcmt', keys)
        jcmwave.jcmt2jcm('./materials.jcmt', keys)
        jcmwave.jcmt2jcm('./project.jcmpt', keys)
        jcmwave.jcmt2jcm('./sources.jcmt', keys)
        jcmwave.jcmt2jcm('./layout.jcmt', keys)
        jcmwave.solve('./project.jcmp')
        
        # Gather Reflected Fourier Modes (Z)
        filename_fourierModes_r = './project_results/fourier_modes_r.jcm';
        fourierModes_r = jcmwave.loadtable(filename_fourierModes_r,format='named')
        powerFlux_r = jcmwave.convert2powerflux(fourierModes_r)

        # Reflected flux in normal direction
        P_s_t = np.sum(powerFlux_r['PowerFluxDensity'][0][:, 2]);
        P_p_t = np.sum(powerFlux_r['PowerFluxDensity'][1][:, 2]); 

        
        filename_MM = './project_results/sm.jcm'

        #print(filename_MM)
        table = jcmwave.loadtable(filename_MM)
        row1 = [
                table['Mueller_xy11'][0], 
                table['Mueller_xy12'][0],
                table['Mueller_xy13'][0],
                table['Mueller_xy14'][0]
            ]
        row2 = [
                table['Mueller_xy21'][0],
                table['Mueller_xy22'][0],
                table['Mueller_xy23'][0],
                table['Mueller_xy24'][0]
            ]
        row3 = [
                table['Mueller_xy31'][0],
                table['Mueller_xy32'][0],
                table['Mueller_xy33'][0],
                table['Mueller_xy34'][0]
            ]
        row4 = [
                table['Mueller_xy41'][0],
                table['Mueller_xy42'][0],
                table['Mueller_xy43'][0],
                table['Mueller_xy44'][0]
            ]
        
        row1 = [1]
        row2 = [2] 
        row3 = [3]
        row4 = [4]
        mm = [row1, row2, row3, row4]
        
        dict[aoi][azi] = {} 
        dict[aoi][azi]["mm"] = mm
        if aoi == 30 and azi == 30:
            print(dict)

        
    toc = time.time() # use time() not clock() on linux system  
    t = toc-tic
    print ("Total runtime for AOI: "+str(aoi)+" aziumth: "+str(azi)+" - %6.4f s" % t)
with open("new_results.json", 'w') as write_file:
    write_file.write(json.dumps(dict, indent=4))
   
