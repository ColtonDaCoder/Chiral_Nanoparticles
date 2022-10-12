import numpy as np
from properpolar import *
import threading

def plot(value, mm):
    AOI = np.arange(20,36,1) #angles of incidence
    azi = np.arange(0,90,2)
    results = []
    azimuth = azi
    for i in range(len(AOI)):
        angle = AOI[i]
        txt = "my_resultsMM_NSL_hexagonal_lattice_radius_37.5_nm_structure_thickness_50_nm_pitch_120_wavelength_2.1e-07nm_" + str(angle) + "AOI.txt"
        lamsi = np.loadtxt(txt, usecols=[3,value])
        results.append(lamsi)



    y = []
    for i in range(len(AOI)):
        y.append(AOI[i])
        
    x = []
    for i in range(len(azimuth)):
        x.append(azimuth[i])

    X,Y = np.meshgrid(np.radians(x),y)

    fig, ax = plt.subplots(figsize=(8,6),subplot_kw=dict(projection='polar'))


    n=len(azimuth)
    m=len(AOI)
    z = np.zeros((m,n))
    print(azimuth)
    for j in range(len(azimuth)):
        for i in range(len(AOI)):
            z[i][j] = results[i][j][1]

    cm = plt.cm.seismic
    fig=ax.pcolormesh(X,Y,z,vmin=-0.8, vmax=0.8, shading='gouraud', antialiased=True, cmap = cm)

    cb = plt.colorbar(fig)
    plt.title(mm,fontsize=20)
    ax.set_rticks([20,30,40,45])
    ax.set_xticks([0,np.pi/12,np.pi/6,np.pi/4,np.pi/3,np.pi/2])
    ax.set_rlim(0, 45)
    ax.set_xlim(0, np.pi/2)
    ax.grid( color = 'gray', linestyle = '--', linewidth = 1 )
    #plot1.axes.get_yaxis().set_visible(False)
    plt.show()

thread1 = threading.Thread(target=plot(7,11))
thread12 = threading.Thread(target=plot(8,12))
thread13 = threading.Thread(target=plot(9,13))
thread14 = threading.Thread(target=plot(10,14))
thread21 = threading.Thread(target=plot(11,21))
thread22 = threading.Thread(target=plot(12,22))
thread23 = threading.Thread(target=plot(13,23))
thread24 = threading.Thread(target=plot(14,24))
thread31 = threading.Thread(target=plot(15,31))
thread32 = threading.Thread(target=plot(16,32))
thread33 = threading.Thread(target=plot(17,33))
thread34 = threading.Thread(target=plot(18,34))
thread41 = threading.Thread(target=plot(19,41))
thread42 = threading.Thread(target=plot(20,42))
thread43 = threading.Thread(target=plot(21,43))
thread44 = threading.Thread(target=plot(22,44))
