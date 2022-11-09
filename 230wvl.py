import numpy as np

file = "my_resultsMM_NSL_hexagonal_lattice_radius_15_nm_structure_thickness_50_nm_pitch_80_azimuth_10_deg_45_deg_AOI_60_nm_aminoacid_.txt"
data = np.loadtxt(file)

print(data[8])
print(data[8][6])
#for i in range(15):
#	data[8][i+7] = data[8][i+7]/data[8][6]

MM12 = data[8][7]
MM13 = data[8][8]
MM14 = data[8][9]
MM21 = data[8][10]
MM22 = data[8][11]
MM23 = data[8][12]
MM24 = data[8][13]
MM31 = data[8][14]
MM32 = data[8][15]
MM33 = data[8][16]
MM34 = data[8][17]
MM41 = data[8][18]
MM42 = data[8][19]
MM43 = data[8][20]
MM44 = data[8][21]


#for i in range(15):	
#	print(data[6][i+7]/data[6][6])
      

DI = np.sqrt(1**2+MM12**2+MM13**2+MM14**2+MM21**2+MM22**2+MM23**2+MM24**2+MM31**2+MM32**2+MM33**2+MM34**2+MM41**2+MM42**2+MM43**2+MM44**2-1**2)/(np.sqrt(3)*1)
print(DI)
