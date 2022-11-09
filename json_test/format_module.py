import json
import numpy as np

class json_file:

    def __init__(self, filename):
        self.data = self.get_json(filename) 
        
    #returns json file of wavelength file
    def get_json(self, filename):
        with open(filename, 'r') as read_file:
            raw = json.load(read_file)
        return raw
    
    #if array is not given return entire mueller matrix
    #else return element indexed by array parameter
    def getMM(self, aoi, azi, element=None):
        mm = self.data[str(aoi)+"AOI"][str(azi)]["mm"]
        if element == None:
            return mm
        else:
            sub = str(element)
            row, column = int(sub[0]),int(sub[1])
            return mm[row-1][column-1]


def tiago_to_json(filename):
    store = {}
    for i in range(46-20):
        aoi = str(i+20) + "AOI" 
        file = filename + aoi + ".txt"
        raw = np.loadtxt(file)
        store[aoi] = {}
        for j in range(31): 
            azi = raw[j][3]
            store[aoi][azi] = {}
            for k in range(16):
                store[aoi][azi]["mm"] = [raw[j][6:10].tolist(), raw[j][10:14].tolist(), raw[j][14:18].tolist(), raw[j][18:22].tolist()]
    return store

def save_json(filename, data): 
    with open(filename, 'w') as write_file:
        write_file.write(json.dumps(data, indent=4))

file = "hex_radius_50_nm_structure_thickness_50_nm_pitch_75_nm_wavelength_2.1e-07nm.json"
tiago = "my_resultsother_rotations_MM_NSL_hexagonal_lattice_radius_50_nm_structure_thickness_50_nm_pitch_175_wavelength_2.1e-07nm_"
save_json("json_test/" + file, tiago_to_json("other_results/" + tiago))
