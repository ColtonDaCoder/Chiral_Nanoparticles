from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import numpy as np
import threading
import multiprocessing

def get_results(value):
    AOI = np.arange(20,36,1) #angles of incidence
    azi = np.arange(0,92,2) #azimuth
    results = []
    azimuth = azi
    for i in range(len(AOI)):
        angle = AOI[i]
        txt = "my_resultsMM_NSL_hexagonal_lattice_radius_37.5_nm_structure_thickness_50_nm_pitch_120_wavelength_2.1e-07nm_" + str(angle) + "AOI.txt"
        lamsi = np.loadtxt("results/" + txt, usecols=[3,value-1])
        results.append(lamsi)

    y = []
    for i in range(len(AOI)):
        y.append(AOI[i])
        
    x = []
    for i in range(len(azimuth)):
        x.append(azimuth[i])

    X,Y = np.meshgrid(np.radians(x),y)

    n = len(azimuth)
    m = len(AOI)
    z = np.zeros((m,n))
    for j in range(len(AOI)):
        for i in range(len(azimuth)): 
            z[j][i] = results[j][i][1]
    z = z/abs(np.amin(z))
    return [X, Y, z] 

def plot(X, Y, Z, mm):

    rTicks = [20,25,30,35] 
    xTicks = [0,np.pi/12,np.pi/6,np.pi/4,np.pi/3,np.pi/2] 

    fig, axis = plt.subplots(4,4,figsize=(10,10),subplot_kw=dict(projection='polar'))
    cm = plt.cm.seismic
    for j in range(4):
        for i in range(4):
            val = i*4+j
            axis[i,j].pcolormesh(X[val],Y[val],Z[val],vmin=-0.8, vmax=0.8, shading='gouraud', antialiased=True, cmap = cm) 
            axis[i,j].set_rticks(rTicks, fontsize=10) #AOI
            axis[i,j].set_xticks(xTicks, fontsize=10)
            axis[i,j].set_rlim(rTicks[0], rTicks[-1])
            axis[i,j].set_xlim(xTicks[0], xTicks[-1]) #azimuth
            axis[i,j].grid( color = 'gray', linestyle = '--', linewidth = 1 )
            axis[i,j].set_title(mm[val],fontsize=20)
    plt.tight_layout(h_pad=1)
    plt.savefig('constant_wvl_210.png')
    plt.show()

if __name__ == '__main__':
    results = [[],[],[],[]]
    for i in range(4):
        for j in range(4):
            data = get_results(7+i*4+j)
            for r in range(3):
                results[r].append(data[r])
            results[3].append(int(str(i+1) + str(j+1)))

    plot(results[0],results[1],results[2], results[3])

