import os
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy import interpolate
import math
import scipy

from scipy.linalg import logm, expm

#Define a function to extract the Mueller Matrix data from a .txt file exported from CompleteEase
def extract_RC2(txt,wmin,wmax,angles):
    
    #Select wvl's of interest to make data processing faster
    
    lamsi = np.loadtxt(txt,skiprows = 1, max_rows=1, usecols = np.arange(1,1042))
    imin = np.where(lamsi == wmin)
    imax = np.where(lamsi == wmax)
    
    rangemin = imin[0][0]+1
    rangemax = imax[0][0]+2

    lams = np.loadtxt(txt,skiprows = 1, max_rows=1, usecols = np.arange(rangemin,rangemax))
    
    n = lams.size
    
    m = angles.size
    
    AOI = np.zeros((m,n)) 
    
    DI = np.zeros((m,n)) 
    
    MM11 = np.ones((m,n))
    MM12 = np.zeros((m,n))
    MM13 = np.zeros((m,n))
    MM14 = np.zeros((m,n))

    MM21 = np.zeros((m,n))
    MM22 = np.zeros((m,n))
    MM23 = np.zeros((m,n))
    MM24 = np.zeros((m,n))

    MM31 = np.zeros((m,n))
    MM32 = np.zeros((m,n))
    MM33 = np.zeros((m,n))
    MM34 = np.zeros((m,n))

    MM41 = np.zeros((m,n))
    MM42 = np.zeros((m,n))
    MM43 = np.zeros((m,n))
    MM44 = np.zeros((m,n))
    
    
    for i in range(m):
        
        j12 = ((m*21)+ 2) + i; j13 = ((m*23)+ 2) + i; j14 = ((m*25)+ 2) + i
        j21 = ((m*27)+ 2) + i; j22 = ((m*29)+ 2) + i; j23 = ((m*31)+ 2) + i; j24 = ((m*33)+ 2) + i
        j31 = ((m*35)+ 2) + i; j32 = ((m*37)+ 2) + i; j33 = ((m*39)+ 2) + i; j34 = ((m*41)+ 2) + i
        j41 = ((m*43)+ 2) + i; j42 = ((m*45)+ 2) + i; j43 = ((m*47)+ 2) + i; j44 = ((m*49)+ 2) + i
        
        AOI [i] = angles[i]
        MM12[i] = np.loadtxt(txt,skiprows = j12, max_rows=1, usecols = np.arange(rangemin+1,rangemax+1))
        MM13[i] = np.loadtxt(txt,skiprows = j13, max_rows=1, usecols = np.arange(rangemin+1,rangemax+1))
        MM14[i] = np.loadtxt(txt,skiprows = j14, max_rows=1, usecols = np.arange(rangemin+1,rangemax+1))
        MM21[i] = np.loadtxt(txt,skiprows = j21, max_rows=1, usecols = np.arange(rangemin+1,rangemax+1))
        MM22[i] = np.loadtxt(txt,skiprows = j22, max_rows=1, usecols = np.arange(rangemin+1,rangemax+1))
        MM23[i] = np.loadtxt(txt,skiprows = j23, max_rows=1, usecols = np.arange(rangemin+1,rangemax+1))
        MM24[i] = np.loadtxt(txt,skiprows = j24, max_rows=1, usecols = np.arange(rangemin+1,rangemax+1))
        MM31[i] = np.loadtxt(txt,skiprows = j31, max_rows=1, usecols = np.arange(rangemin+1,rangemax+1))
        MM32[i] = np.loadtxt(txt,skiprows = j32, max_rows=1, usecols = np.arange(rangemin+1,rangemax+1))
        MM33[i] = np.loadtxt(txt,skiprows = j33, max_rows=1, usecols = np.arange(rangemin+1,rangemax+1))
        MM34[i] = np.loadtxt(txt,skiprows = j34, max_rows=1, usecols = np.arange(rangemin+1,rangemax+1))
        MM41[i] = np.loadtxt(txt,skiprows = j41, max_rows=1, usecols = np.arange(rangemin+1,rangemax+1))
        MM42[i] = np.loadtxt(txt,skiprows = j42, max_rows=1, usecols = np.arange(rangemin+1,rangemax+1))
        MM43[i] = np.loadtxt(txt,skiprows = j43, max_rows=1, usecols = np.arange(rangemin+1,rangemax+1))
        MM44[i] = np.loadtxt(txt,skiprows = j44, max_rows=1, usecols = np.arange(rangemin+1,rangemax+1))
        DI[i] = np.sqrt(1**2+MM12[i]**2+MM13[i]**2+MM14[i]**2+MM21[i]**2+MM22[i]**2+MM23[i]**2+MM24[i]**2+MM31[i]**2+MM32[i]**2+MM33[i]**2+MM34[i]**2+MM41[i]**2+MM42[i]**2+MM43[i]**2+MM44[i]**2-1**2)/(np.sqrt(3)*1)
    MM=[lams,MM11,MM12,MM13,MM14,MM21,MM22,MM23,MM24,MM31,MM32,MM33,MM34,MM41,MM42,MM43,MM44,AOI,DI]

    return MM

def Cloude_Decomposition(MM,angles):
    [lams,MM11,MM12,MM13,MM14,MM21,MM22,MM23,MM24,MM31,MM32,MM33,MM34,MM41,MM42,MM43,MM44,AOI,DI] = MM
    
    n = lams.size
    
    m = angles.size
    
    dMM11 = np.zeros((m,n))
    dMM12 = np.zeros((m,n))
    dMM13 = np.zeros((m,n))
    dMM14 = np.zeros((m,n))
    dMM21 = np.zeros((m,n))
    dMM22 = np.zeros((m,n))
    dMM23 = np.zeros((m,n))
    dMM24 = np.zeros((m,n))
    dMM31 = np.zeros((m,n))
    dMM32 = np.zeros((m,n))
    dMM33 = np.zeros((m,n))
    dMM34 = np.zeros((m,n))
    dMM41 = np.zeros((m,n))
    dMM42 = np.zeros((m,n))
    dMM43 = np.zeros((m,n))
    dMM44 = np.zeros((m,n))
    
    #Cloude Decomposition
        
    e1 = np.zeros((m,n))
    e2 = np.zeros((m,n))
    e3 = np.zeros((m,n))
    e4 = np.zeros((m,n))
    
    h11 = np.zeros((m,n),dtype=complex)
    h12 = np.zeros((m,n),dtype=complex)
    h13 = np.zeros((m,n),dtype=complex)
    h14 = np.zeros((m,n),dtype=complex)
    
    h21 = np.zeros((m,n),dtype=complex)
    h22 = np.zeros((m,n),dtype=complex)
    h23 = np.zeros((m,n),dtype=complex)
    h24 = np.zeros((m,n),dtype=complex)
    
    h31 = np.zeros((m,n),dtype=complex)
    h32 = np.zeros((m,n),dtype=complex)
    h33 = np.zeros((m,n),dtype=complex)
    h34 = np.zeros((m,n),dtype=complex)
    
    h41 = np.zeros((m,n),dtype=complex)
    h42 = np.zeros((m,n),dtype=complex)
    h43 = np.zeros((m,n),dtype=complex)
    h44 = np.zeros((m,n),dtype=complex)
    
    for j in range(m):
        for i in range (n):
        
            h11 = ((MM11[j][i]+MM22[j][i]+MM33[j][i]+MM44[j][i])/4)
            h12 = ((MM12[j][i] + MM21[j][i] - 1j*MM34[j][i] + 1j*MM43[j][i])/4)
            h13 = ((MM13[j][i] + MM31[j][i] + 1j*MM24[j][i] - 1j*MM42[j][i])/4)
            h14 = ((MM14[j][i] - 1j*MM23[j][i] + 1j*MM32[j][i] + MM41[j][i])/4)

            h21 = ((MM12[j][i] + MM21[j][i] + 1j*MM34[j][i] - 1j*MM43[j][i])/4)
            h22 = ((MM11[j][i] + MM22[j][i] - MM33[j][i] - MM44[j][i])/4)
            h23 = ((1j*MM14[j][i] + MM23[j][i] + MM32[j][i] - 1j*MM41[j][i])/4)
            h24 = ((- 1j*MM13[j][i] + 1j*MM31[j][i] + MM24[j][i] + MM42[j][i])/4)

            h31 = ((MM13[j][i] + MM31[j][i] - 1j*MM24[j][i] + 1j*MM42[j][i])/4)
            h32 = ((- 1j*MM14[j][i]+ MM23[j][i] + MM32[j][i] + 1j*MM41[j][i])/4)
            h33 = ((MM11[j][i] - MM22[j][i] + MM33[j][i] - MM44[j][i])/4)
            h34 = ((1j*MM12[j][i] - 1j*MM21[j][i] + MM34[j][i] + MM43[j][i])/4)

            h41 = ((MM14[j][i] + 1j*MM23[j][i] - 1j*MM32[j][i] + MM41[j][i])/4)
            h42 = ((1j*MM13[j][i] - 1j*MM31[j][i] + MM24[j][i] + MM42[j][i])/4)
            h43 = ((- 1j*MM12[j][i] + 1j*MM21[j][i] + MM34[j][i] + MM43[j][i])/4)
            h44 = ((MM11[j][i] - MM22[j][i] - MM33[j][i] + MM44[j][i])/4)
            
            #Create coherency matrix
            H = np.array([[h11,h12,h13,h14], [h21,h22,h23,h24], [h31,h32,h33,h34], [h41,h42,h43,h44]])
            w,v = np.linalg.eigh(H)
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
            
            #Normalize dMM
            dMM11[j][i] = M[0,0]/M[0,0]
            dMM12[j][i] = M[0,1]/M[0,0]
            dMM13[j][i] = M[0,2]/M[0,0]
            dMM14[j][i] = M[0,3]/M[0,0]
            dMM21[j][i] = M[1,0]/M[0,0]
            dMM22[j][i] = M[1,1]/M[0,0]
            dMM23[j][i] = M[1,2]/M[0,0]
            dMM24[j][i] = M[1,3]/M[0,0]
            dMM31[j][i] = M[2,0]/M[0,0]
            dMM32[j][i] = M[2,1]/M[0,0]
            dMM33[j][i] = M[2,2]/M[0,0]
            dMM34[j][i] = M[2,3]/M[0,0]
            dMM41[j][i] = M[3,0]/M[0,0]
            dMM42[j][i] = M[3,1]/M[0,0]
            dMM43[j][i] = M[3,2]/M[0,0]
            dMM44[j][i] = M[3,3]/M[0,0]
            e1[j][i] = w[3]
            e2[j][i] = w[0]
            e3[j][i] = w[1]
            e4[j][i] = w[2]


    
    dMM=[lams,dMM11,dMM12,dMM13,dMM14,dMM21,dMM22,dMM23,dMM24,dMM31,dMM32,dMM33,dMM34,dMM41,dMM42,dMM43,dMM44,e1,e2,e3,e4,AOI]
    return dMM

def D_MM(MM,dMM,angles):
    [lams,MM11,MM12,MM13,MM14,MM21,MM22,MM23,MM24,MM31,MM32,MM33,MM34,MM41,MM42,MM43,MM44,AOI,DI] = MM
    [lams,dMM11,dMM12,dMM13,dMM14,dMM21,dMM22,dMM23,dMM24,dMM31,dMM32,dMM33,dMM34,dMM41,dMM42,dMM43,dMM44,e1,e2,e3,e4,AOI]=dMM
    
    n=len(lams)
    m = angles.size
    
    L11 = np.zeros((m,n))
    L12 = np.zeros((m,n))
    L13 = np.zeros((m,n))
    L14 = np.zeros((m,n))
    L21 = np.zeros((m,n))
    L22 = np.zeros((m,n))
    L23 = np.zeros((m,n))
    L24 = np.zeros((m,n))
    L31 = np.zeros((m,n))
    L32 = np.zeros((m,n))
    L33 = np.zeros((m,n))
    L34 = np.zeros((m,n))
    L41 = np.zeros((m,n))
    L42 = np.zeros((m,n))
    L43 = np.zeros((m,n))
    L44 = np.zeros((m,n))
    
    G = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0], [0,0,0,-1]]) 
    
    for j in range(m):
        for i in range (n):
        
            M =  np.array([[dMM11[j][i],dMM12[j][i],dMM13[j][i],dMM14[j][i]], [dMM21[j][i],dMM22[j][i],dMM23[j][i],dMM24[j][i]], [dMM31[j][i],dMM32[j][i],dMM33[j][i],dMM34[j][i]], [dMM41[j][i],dMM42[j][i],dMM43[j][i],dMM44[j][i]]])

            dL = logm(M)

            dLT = np.transpose(dL)

            L = 0.5*(dL - G@dLT@G)


            L11[j][i] = L[0,0]
            L12[j][i] = L[0,1]
            L13[j][i] = L[0,2]
            L14[j][i] = L[0,3]
            L21[j][i] = L[1,0]
            L22[j][i] = L[1,1]
            L23[j][i] = L[1,2]
            L24[j][i] = L[1,3]
            L31[j][i] = L[2,0]
            L32[j][i] = L[2,1]
            L33[j][i] = L[2,2]
            L34[j][i] = L[2,3]
            L41[j][i] = L[3,0]
            L42[j][i] = L[3,1]
            L43[j][i] = L[3,2]
            L44[j][i] = L[3,3]
    
    DMM=[lams,L11,L12,L13,L14,L21,L22,L23,L24,L31,L32,L33,L34,L41,L42,L43,L44,AOI,DI,dMM11,dMM12,dMM13,dMM14,dMM21,dMM22,dMM23,dMM24,dMM31,dMM32,dMM33,dMM34,dMM41,dMM42,dMM43,dMM44]
    return DMM
