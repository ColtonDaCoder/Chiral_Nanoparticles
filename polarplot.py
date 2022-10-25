from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import numpy as np
import threading
import multiprocessing

def get_results(value, radius_range, txt):
    radius = radius_range 
    azimuth = np.arange(0,92,2) #azimuth
    results = []
    for i in range(len(radius)):
        lamsi = np.loadtxt("results/" + txt[0] + str(radius[i]) + txt[1], usecols=[3,value-1])
        results.append(lamsi)

    y = []
    for i in range(len(radius)):
        y.append(radius[i])
        
    x = []
    for i in range(len(azimuth)):
        x.append(azimuth[i])

    X,Y = np.meshgrid(np.radians(x),y)

    n = len(azimuth)
    m = len(radius)
    z = np.zeros((m,n))
    for j in range(len(radius)):
        for i in range(len(azimuth)): 
            z[j][i] = results[j][i][1]
    #z = z/abs(np.amax(abs(z)))
    return [X, Y, z] 

def plot(X, Y, Z, DECOMP=False):

    rTicks = [20,25,30,35] 
    xTicks = [0,np.pi/12,np.pi/6,np.pi/4,np.pi/3,np.pi/2] 

    fig, axis = plt.subplots(4,4,figsize=(10,10),subplot_kw=dict(projection='polar'))
    cm = plt.cm.hot
    for j in range(4):
        for i in range(4):
            val = i*4+j
            Zmin = np.amin(Z[val])
            Zmax = np.amax(Z[val])
            cb = axis[i,j].pcolormesh(X[val],Y[val],Z[val],vmin=Zmin, vmax=Zmax, shading='gouraud', antialiased=True, cmap = cm) 
            axis[i,j].set_rticks(rTicks, fontsize=10) #AOI
            axis[i,j].set_xticks(xTicks, fontsize=10)
            axis[i,j].set_rlim(rTicks[0], rTicks[-1])
            axis[i,j].set_xlim(xTicks[0], xTicks[-1]) #azimuth
            axis[i,j].grid( color = 'gray', linestyle = '--', linewidth = 1 )
            title = str(j+1) + str(i+1)
            if DECOMP and (i == 0 and j == 0):
                title = "DI"
            fig.colorbar(cb,ax=axis[i,j],pad=0.2)

            axis[i,j].set_title(title,fontsize=20)
    plt.tight_layout(h_pad=1,w_pad=3)
    plt.savefig('constant_wvl_210.png')
    plt.show()

if __name__ == '__main__':
    results = [[],[],[]]
    #radius is AOI
    radius_range = np.arange(20,36,1)
    txt = ["my_resultsMM_NSL_hexagonal_lattice_radius_37.5_nm_structure_thickness_50_nm_pitch_120_wavelength_2.1e-07nm_", "AOI.txt"]

    for i in range(4):
        for j in range(4):
            data = get_results(7+i*4+j,radius_range,txt)
            for r in range(3):
                results[r].append(data[r])
    plot(results[0],results[1],results[2])

