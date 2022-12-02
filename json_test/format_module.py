import json
import matplotlib.pyplot as plt
import numpy as np
import copy

class json_file:

    def __init__(self, filename, is_new=False):
        self.filename = filename
        if is_new:
            self.data = {}
        else:
            self.data = self.get_json(self.filename) 

    #returns json file of wavelength file
    def get_json(self, filename):
        with open(filename, 'r') as read_file:
            raw = json.load(read_file)
        return raw
 
    #if array is not given return entire mueller matrix
    #else return element indexed by array parameter
    def getMM(self, aoi, azi, element=None):
        mm = self.get_data()[str(aoi)][str(azi)]["mm"]
        if element == None:
            return mm
        else:
            sub = str(element)
            row, column = int(sub[0]),int(sub[1])
            return mm[row-1][column-1]

    def getdMM(self, aoi, azi, element=None):
        dMM = self.get_data()[str(aoi)][str(azi)]["dMM"]
        if element == None:
            return dMM
        else:
            sub = str(element)
            row, column = int(sub[0]),int(sub[1])
            return dMM[row-1][column-1]



    #using tiago DI
    def add_DI(self):        
        new_data = self.get_data()
        for aoi in self.get_data():
            for azi in self.data[aoi]:
                mm = np.asarray(self.getMM(aoi,azi)).flatten()
                di = np.sqrt(abs(sum(i**2 for i in mm) - mm[0]**2 ))/(np.sqrt(3)*mm[0])
                new_data[aoi][azi]["DI"] = di
        self.save_json(self.filename,new_data)

    def get_data(self):
        return copy.deepcopy(self.data)

    def getMM_col(self, element):
        new_data = self.get_data()
        for aoi in new_data:
            for azi in new_data[aoi]:
                new_data[aoi][azi]["dMM"] = self.getdMM(aoi,azi,element)
        return new_data

    def add_DMM(self):
        new_data = self.get_data()
        for aoi in self.get_data():
            for azi in self.data[aoi]:
                mm = np.asarray(self.getMM(aoi,azi))
                new_data[aoi][azi]["dMM"] = self.cloude_decomp(mm).tolist()
        self.save_json(self.filename,new_data) 

    #cloude decomposition matrix
    def cloude_decomp(self, mm):
        #pauli matrices
        p = [
                [
                    [1,0],
                    [0,1]
                ],
                [
                    [1,0],
                    [0,-1]
                ],
                [
                    [0,1],
                    [1,0]
                ],
                [
                    [0,-1j],
                    [1j,0]
                ]
            ]
        A = np.array([np.conj(i).flatten() for i in p])
        #coherency matrix = (1/4) A(pauli(sub i) kronecker pauli(sub j))A^-1
        sigma = 0
        for i in range(4):
            for j in range(4):
                sigma = sigma + mm[i][j]*(A@np.kron(p[i],np.conj(p[j]))@np.linalg.inv(A))
        C = (1/4) * sigma

        w,v = np.linalg.eigh(C)
        Ψ1=np.transpose(v[:, 3])
        j11=Ψ1[0] + Ψ1[1]
        j12=Ψ1[2] - 1j*Ψ1[3]
        j21=Ψ1[2] + 1j*Ψ1[3]
        j22=Ψ1[0]-Ψ1[1]
        J=np.array([[j11,j12],[j21,j22]])
        A = (1/np.sqrt(2))*np.array([[1, 0, 0, 1], [1, 0, 0, -1], [0, 1, 1, 0], [0, 1j, -1j, 0]])
        Ainv = np.linalg.inv(A)
        Jconj=np.matrix.conjugate(J)
        M = A@(np.kron(J,Jconj))@Ainv
        M = M.real
        return M 

    def save_json(self, filename, data): 
        with open(filename, 'w') as write_file:
            write_file.write(json.dumps(data, indent=4))

def save_json(filename, data): 
    with open(filename, 'w') as write_file:
        write_file.write(json.dumps(data, indent=4))

def tiago_to_json(filename):
    store = {}
    for i in range(35-20):
        aoi = str(i+20) + "AOI" 
        file = filename + aoi + ".txt"
        raw = np.loadtxt(file)
        store[aoi] = {}
        for j in range(46): 
            azi = raw[j][3]
            store[aoi][azi] = {}
            for k in range(16):
                store[aoi][azi]["mm"] = [raw[j][6:10].tolist(), raw[j][10:14].tolist(), raw[j][14:18].tolist(), raw[j][18:22].tolist()]
    return store

def convert_to_plot(file):
    all_data = json_file("json_test/"+file).data
    y = [int(aoi[0:2]) for aoi in all_data]
    x = [float(azi) for azi in all_data["20AOI"]]
    base_x,base_y = np.meshgrid(np.radians(x),y)
    X = [base_x for i in range(16)]
    Y = [base_y for i in range(16)] 
    elements = [11,12,13,14,21,22,23,24,31,32,33,34,41,42,43,44]
    Z=[[ [] for aoi in all_data ] for e in elements]
    for i, aoi in enumerate(all_data): 
        for azi in all_data[aoi]:
            dMM = all_data[aoi][azi]["dMM"]
            for index, element in enumerate(elements):
                sub = str(element)
                Z[index][i].append(dMM[int(sub[0])-1][int(sub[1])-1])
    return X,Y,Z

def plot(file, DECOMP=False):
    X,Y,Z = convert_to_plot(file)
    #X = azimuth (xticks)
    #Y = AOI (rticks)
    #Z = dMM
    rTicks = [20,25,30,35] 
    xTicks = [0,np.pi/12,np.pi/6,np.pi/4,np.pi/3,np.pi/2] 

    fig, axis = plt.subplots(4,4,figsize=(10,10),subplot_kw=dict(projection='polar'))
    cm = plt.cm.seismic
    for j in range(4):
        for i in range(4):
            val = i*4+j
            Zmin = np.amin(Z[val])
            Zmax = np.amax(Z[val])
            if DECOMP and (i == 0 and j == 0):
                Zmin = 0.9
                Zmax = 1.1
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

file = "hex_radius_37.5_nm_structure_thickness_50_nm_pitch_120_wavelength_2.1e-07nm.json"
plot(file, DECOMP=True)
    
#file = "hex_radius_50_nm_structure_thickness_50_nm_pitch_75_nm_wavelength_2.1e-07nm.json"
#tiago = "my_resultsother_rotations_MM_NSL_hexagonal_lattice_radius_50_nm_structure_thickness_50_nm_pitch_175_wavelength_2.1e-07nm_"
#tiago = "my_resultsMM_NSL_hexagonal_lattice_radius_37.5_nm_structure_thickness_50_nm_pitch_120_wavelength_2.1e-07nm_"
#save_json("json_test/" + file, tiago_to_json("results/" + tiago))




