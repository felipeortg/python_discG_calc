#!/usr/bin/env python
# coding: utf-8

# # Discontinuities of $\mathcal{I}_{\mathcal{A}}^\nu$

# In[1]:


import numpy as np

from scipy import integrate
from scipy import optimize
from scipy import special

import matplotlib.pyplot as plt


# In[2]:


I = complex(0,1.)

m1 = 1.
m2 = 1.


# In[3]:


# Define useful functions for the calculation with Feynman parameters
def y_noie(x, Lam1, Lam2, pm, q2, si, sf):
    
    x = complex(x, 0)
    
    if pm == 'p':
        pm = 1
    else:
        pm = -1 
              
    AA = 1 + (Lam2**2 - Lam1**2 + x * (q2 - sf - si)) / si
    
    BB = -4 * (Lam2**2 - x * (Lam2**2 - Lam1**2) - x * (1 - x) * sf) / si
    
    return 0.5 * (AA + pm * np.sqrt(AA**2 + BB))

    
# Generalized antider of the 1 pole integral
def antiderlog(ymin, ymax, pole, pm):
    
    if pole == ymin or pole == ymax:
        return float('nan')

    if pm == 'p':
        pm = +1
    else:
        pm = -1

    # Case of the non-imag pole
    if np.angle(pole)==0 or np.abs(np.angle(pole)) == np.pi:
        rpole = np.real(pole)

        # Pole in the domain: PV +/- I pi 
        if (rpole - ymin) * (ymax - rpole) > 0:
            repart = np.log(np.abs((ymax - rpole) / (ymin - rpole))) #abs ensures the argument is real

            impart = pm * I * np.pi

            return repart + impart

        else: # no pole in the domain of integration
            repart = np.log(np.abs((ymax - rpole) / (ymin - rpole)))

            return repart

    # Case of imaginary pole   
    else:
        rpole = np.real(pole)
        ipole = np.imag(pole)

        repart = 0.5 * np.log(((ymax - rpole)**2 + ipole**2)/((ymin - rpole)**2 + ipole**2))

        impart = I * (np.arctan((ymax - rpole) / ipole) - np.arctan((ymin - rpole) / ipole))
        
        # Checking if abs substitution works
#         if debug:
#             repart = np.log(np.abs((ymax - pole) / (ymin - pole)))

        return repart + impart


def F1(x, Lams, kins):
    
    x = complex(x, 0)
    
    q2 = kins[0]
    si = kins[1]
    sf = kins[2]
    
    yp = y_noie(x, Lams[0], Lams[1], 'p', q2, si, sf)
    ym = y_noie(x, Lams[0], Lams[1], 'm', q2, si, sf)
    
    
    ffcoef = 1. / (si * (4 * np.pi)**2 * (yp - ym))
  

    return ffcoef * (antiderlog(0, 1 - x, yp, 'p') - antiderlog(0, 1 - x, ym, 'm'))

def F2(x, Lams, kins):
    
    x = complex(x, 0)
    
    q2 = kins[0]
    si = kins[1]
    sf = kins[2]
    
    yp = y_noie(x, Lams[0], Lams[1], 'p', q2, si, sf)
    ym = y_noie(x, Lams[0], Lams[1], 'm', q2, si, sf)
    
    
    ffcoef = 1. / (si * (4 * np.pi)**2 * (yp - ym))
  

    return ffcoef * (yp * antiderlog(0, 1 - x, yp, 'p') - ym * antiderlog(0, 1 - x, ym, 'm'))


# In[4]:


# Functions for calculation with Cutkosky

def dot_4vec(quadvec1,quadvec2):
    return (quadvec1[0]*quadvec2[0]) - sum([(quadvec1[ii])*quadvec2[ii] for ii in range(1,4)])

def square_4vec(quadvec):
    return dot_4vec(quadvec,quadvec)

def qq2(P4v, ma, mb):
    ss = square_4vec(P4v)
    
    return (ss - 2*(ma**2 + mb**2) + (ma**2 - mb**2)**2/ss)/4

# Define boost matrix
# Where boost(p4vector).p4vector = (E^*, vec(0))
def boost(p4vector):
    # velocity = |vec(p)|/E
    beta = np.sqrt(np.dot(p4vector[1:4],p4vector[1:4]))/p4vector[0]

    # no velocity
    if beta==0:
        return np.identity(4)

    gamma = 1/np.sqrt(1-beta**2)
    norm = p4vector[1:4]/(beta*p4vector[0]) # normed momentum

    resul= np.array([
        [gamma, -gamma*norm[0]*beta, -gamma*norm[1]*beta, -gamma*norm[2]*beta],
        [-gamma*norm[0]*beta, 1+(gamma-1)*norm[0]**2, (gamma-1)*norm[0]*norm[1], (gamma-1)*norm[0]*norm[2]],
        [-gamma*norm[1]*beta, (gamma-1)*norm[0]*norm[1], 1+(gamma-1)*norm[1]**2, (gamma-1)*norm[1]*norm[2]],
        [-gamma*norm[2]*beta, (gamma-1)*norm[0]*norm[2], (gamma-1)*norm[1]*norm[2], 1+(gamma-1)*norm[2]**2]])

    return resul


# In[5]:


# Functions for calculation with Cutkosky
def zz(P4vCM, P4vNOCM, ma, mb):
    qCM2 = qq2(P4vCM, ma, mb)
    
    # Get the correct branch cut
    if qCM2 < 0:
        qCM = I * np.sqrt(-qCM2)
    else:
        qCM = np.sqrt(qCM2)
    
    omegaqbCM = np.sqrt(qCM2 + mb**2)
    omegaqaCM = np.sqrt(qCM2 + ma**2)
    
    Lamb = boost(P4vCM)
    
    P4v = np.dot(Lamb, P4vNOCM)
#     print(Lamb,P4v)
    
    P3m2 = sum([(P4v[ii])**2 for ii in range(1,4)])
    
    P3m = np.sqrt(P3m2)
    
    return (P3m2 + omegaqbCM**2 - (P4v[0]-omegaqaCM)**2)/(2*P3m*qCM)

def zzst(Efst, P3_f, Eist, P3_i, ma, mb):
    P3m2f = sum([(P3_f[ii])**2 for ii in range(3)])
    P3m2i = sum([(P3_i[ii])**2 for ii in range(3)])
    
    P4f = np.array([np.sqrt(Efst**2 + P3m2f),P3_f[0],P3_f[1],P3_f[2]])
    
    P4i = np.array([np.sqrt(Eist**2 + P3m2i),P3_i[0],P3_i[1],P3_i[2]])
                   
    return zz(P4f, P4i, ma, mb)

def zzstcoef(Efst, P3_f, Eist, P3_i, ma, mb):
    P3m2f = sum([(P3_f[ii])**2 for ii in range(3)])
    P3m2i = sum([(P3_i[ii])**2 for ii in range(3)])
    
    P4f = np.array([np.sqrt(Efst**2 + P3m2f),P3_f[0],P3_f[1],P3_f[2]])
    
    P4i = np.array([np.sqrt(Eist**2 + P3m2i),P3_i[0],P3_i[1],P3_i[2]])
    
    qCM2 = qq2(P4f, ma, mb)
    
    # Get the correct branch cut
    if qCM2 < 0:
        qCM = I * np.sqrt(-qCM2)
    else:
        qCM = np.sqrt(qCM2)
    
                   
    return qCM * zz(P4f, P4i, ma, mb)
    
def DiscIAnu_PineqPf(nu, Efst, P3_f, Eist, P3_i, ma, mb, fiorboth):
    P3m2f = sum([(P3_f[ii])**2 for ii in range(3)])
    P3m2i = sum([(P3_i[ii])**2 for ii in range(3)])
    
    P4f = np.array([np.sqrt(Efst**2 + P3m2f),P3_f[0],P3_f[1],P3_f[2]])
    
    P4i = np.array([np.sqrt(Eist**2 + P3m2i),P3_i[0],P3_i[1],P3_i[2]])
    
    P4fm = np.array([np.sqrt(Efst**2 + P3m2f),-P3_f[0],-P3_f[1],-P3_f[2]])
    
    P4im = np.array([np.sqrt(Eist**2 + P3m2i),-P3_i[0],-P3_i[1],-P3_i[2]])
    
    
    denom = 16 * np.sqrt(dot_4vec(P4f,P4i)**2 - square_4vec(P4f) * square_4vec(P4i) )
    
    if denom == 0:
        return float('nan')
             

    Disc = 0
    
    if Efst >= (ma+mb) and (fiorboth != 'i'):
        zzf = zz(P4f,P4i,ma,mb)
        
        # Some extra care is needed at threshold
        if Efst == (ma+mb): 
            lognumf = 1
            logdenomf = 1
        else:
            lognumf = (1 + zzf) 
            logdenomf = (1 - zzf)

        if lognumf*logdenomf == 0:
            return float('nan')        

        qCM2 = qq2(P4f, ma, mb)
        omegaqaCM = np.sqrt(qCM2 + ma**2)
        # Get the correct branch cut
        if qCM2 < 0:
            qCM = I * np.sqrt(-qCM2)
        else:
            qCM = np.sqrt(qCM2)
            
        Lamf = boost(P4f) 
        Lammf = boost(P4fm)  
        P3if = np.dot(Lamf,P4i)[1:4]
        P3ifmag = np.sqrt(np.sum([P3if[ii]**2 for ii in range(3)]))
        hatP3if = P3if/P3ifmag
#         print(hatP3if)
        
        imagpart = 0
         
        if 1 + zzf > 0:
            imagpart += 1
        if zzf - 1 > 0:
            imagpart -= 1
        
        nonregQ = np.log(np.abs(lognumf/logdenomf))/np.pi + I * imagpart 
        
        temp = Lammf[nu,0] * omegaqaCM * nonregQ
#         print(nonregQ)
    
        # Some extra care is needed at threshold
        if Efst == (ma+mb):  
            spat = 0
        else:
            boostspat = np.sum([Lammf[nu,ii+1] * hatP3if[ii] for ii in range(3)])
            spat = boostspat * qCM * ( zzf * nonregQ - 2/np.pi)
        
  
        Disc+= I/denom * (temp + spat)
        
    if Eist >= (ma+mb) and (fiorboth != 'f'):
        zzi = zz(P4i,P4f,ma,mb)
        
        # Some extra care is needed at threshold
        if Eist == (ma+mb):
            lognumi = 1 
            logdenomi = 1
        else:
            lognumi = (1 + zzi) 
            logdenomi = (1 - zzi)
                  
        if lognumi*logdenomi == 0:
            return float('nan')
        
        qCM2 = qq2(P4i, ma, mb)
        omegaqaCM = np.sqrt(qCM2 + ma**2)
        # Get the correct branch cut
        if qCM2 < 0:
            qCM = I * np.sqrt(-qCM2)
        else:
            qCM = np.sqrt(qCM2)
            
        Lami = boost(P4i) 
        Lammi = boost(P4im)  
        P3fi = np.dot(Lami,P4f)[1:4]
        P3fimag = np.sqrt(np.sum([P3fi[ii]**2 for ii in range(3)]))
        hatP3fi = P3fi/P3fimag
#         print(hatP3fi)
        
        imagpart = 0
         
        # Conjugate the second leg of one of the diagrams  
        if fiorboth == 'both':
            if 1 + zzi > 0:
                imagpart -= 1
            if zzi - 1 > 0:
                imagpart += 1
        else:
            if 1 + zzi > 0:
                imagpart += 1
            if zzi - 1 > 0:
                imagpart -= 1
        
#         print(imagpart)
        nonregQ = np.log(np.abs(lognumi/logdenomi))/np.pi + I * imagpart 

        temp = Lammi[nu,0] * omegaqaCM * nonregQ
        
        # Some extra care is needed at threshold
        if Eist == (ma+mb):  
            spat = 0
        else:
            boostspat = np.sum([Lammi[nu,ii+1] * hatP3fi[ii] for ii in range(3)])
            spat = boostspat * qCM * ( zzi * nonregQ - 2/np.pi)
        
#         print(qCM , zzi , nonregQ)
#         print(spat)
#         print(temp+spat)
#         print(denom)
        
            
        Disc+= I/denom * (temp + spat)
        
    return Disc    


# ## Quick checks of the Cutkosky things

# In[70]:


DiscIAnu_PineqPf(0, 2.5, [0,0,0], 2.05, [0,0,-(2 * np.pi)/6.], m1, m2, 'f')
# (-0.0064164372520079775+0.02178372208892361j)


# In[71]:


[DiscIAnu_PineqPf(0, 2.2, [0,0,0], 2.05, [0,0,-(2 * np.pi)/6.], m1, m2, 'i') ,
 DiscIAnu_PineqPf(0, 2.2, [0,0,0], 2.05, [0,0,(2 * np.pi)/6.], m1, m2, 'i') ,
DiscIAnu_PineqPf(0, 2.22, [0,0,0], 2.05, [0,0,(2 * np.pi)/6.], m1, m2, 'i') ]


# In[72]:


DiscIAnu_PineqPf(0, 2.05, [0,0,(2 * np.pi)/6.], 2.2, [0,0,0], m1, m2, 'both') 
#i irrespective of sign (-0.029841551829730386+0.020024474480493035j)
#f irrespective of sign  (-0.029999545582961133+0.009916850651500924j)
#both irrespective of sign -0.00015799375323074724+0.02994132513199396j), 0.03881166059018775j


# ### Note that the imaginary part exactly cancels

# In[73]:


PFsq = ((2 * np.pi)/6.)**2
PIsq = 0*((2 * np.pi)/6.)**2

EI = [2.05]

EF = np.linspace(1.9,5,num=500)
print(len(EF))
IA0 = np.zeros(len(EF))
IA0i = np.zeros(len(EF))
IA0disc = np.zeros(len(EF))
IA0disci = np.zeros(len(EF))
IA0discb = np.zeros(len(EF))
IA0discbi = np.zeros(len(EF))


for ei in EI:
    for nn, ef in enumerate(EF):
        
        if nn == 30:
            print(0, ef, [0,0,np.sqrt(PFsq)], ei, [0,0,-np.sqrt(PIsq)], m1, m2, 'both')
        
        IA0disctot = DiscIAnu_PineqPf(0, ef, [0,0,np.sqrt(PFsq)], ei, [0,0,-np.sqrt(PIsq)], m1, m2, 'i')
        
        IA0[nn] = np.real(IA0disctot)
        IA0i[nn] = np.imag(IA0disctot)
        
        IA0disctot = DiscIAnu_PineqPf(0, ef, [0,0,np.sqrt(PFsq)], ei, [0,0,-np.sqrt(PIsq)], m1, m2, 'f')
        
        IA0disc[nn] = np.real(IA0disctot)
        IA0disci[nn] = np.imag(IA0disctot)
        
        IA0disctot = DiscIAnu_PineqPf(0, ef, [0,0,np.sqrt(PFsq)], ei, [0,0,-np.sqrt(PIsq)], m1, m2, 'both')
        
        IA0discb[nn] = np.real(IA0disctot)
        IA0discbi[nn] = np.imag(IA0disctot)
        
    plt.figure(1,figsize=(10, 6))
    
    plt.axhline(0,ls='--',color='k')
    
    plt.plot(EF,IA0disc, label="Discf, " + r'$E_i^\star=$'+str(ei))
    
    plt.plot(EF,IA0, label="Disci, " +r'$E_i^\star=$'+str(ei))
    
    plt.plot(EF,IA0discb, label="Disc, " +r'$E_i^\star=$'+str(ei))

    plt.figure(2,figsize=(10, 6))
    
    plt.plot(EF,IA0disci, label="Discf, " + r'$E_i^\star=$'+ str(ei))
    
    plt.plot(EF,IA0i, label="Disci, " + r'$E_i^\star=$'+ str(ei))
    
    plt.plot(EF,IA0discbi, label="Disc, " +r'$E_i^\star=$'+str(ei))
    
plt.figure(1)
plt.title(r'$I_{A0}$'+" Real part")
plt.legend()


plt.figure(2)
plt.title(r'$I_{A0}$'+" Imag part")
plt.legend()


# In[44]:


## Disc version to play
def DiscIAnu_PineqPf_play(nu, Efst, P3_f, Eist, P3_i, ma, mb, fiorboth, tempspatorboth):
    P3m2f = sum([(P3_f[ii])**2 for ii in range(3)])
    P3m2i = sum([(P3_i[ii])**2 for ii in range(3)])
    
    P4f = np.array([np.sqrt(Efst**2 + P3m2f),P3_f[0],P3_f[1],P3_f[2]])
    
    P4i = np.array([np.sqrt(Eist**2 + P3m2i),P3_i[0],P3_i[1],P3_i[2]])
    
    P4fm = np.array([np.sqrt(Efst**2 + P3m2f),-P3_f[0],-P3_f[1],-P3_f[2]])
    
    P4im = np.array([np.sqrt(Eist**2 + P3m2i),-P3_i[0],-P3_i[1],-P3_i[2]])
    
    
    denom = 16 * np.sqrt(dot_4vec(P4f,P4i)**2 - square_4vec(P4f) * square_4vec(P4i) )
    
    if denom == 0:
        return float('nan')
             
    
#     Lammf = boost(P4fm)
#     print(Lammf[nu,:])
#     Lammi = boost(P4im)
#     print(Lammi[nu,:])
    
    Disc = 0
    
    if Efst >= (ma+mb) and (fiorboth != 'i'):
        zzf = zz(P4f,P4i,ma,mb)
        lognumf = (1 + zzf) 
        logdenomf = (1 - zzf)
        if lognumf*logdenomf == 0:
            return float('nan')        

        qCM2 = qq2(P4f, ma, mb)
        omegaqaCM = np.sqrt(qCM2 + ma**2)
        # Get the correct branch cut
        if qCM2 < 0:
            qCM = I * np.sqrt(-qCM2)
        else:
            qCM = np.sqrt(qCM2)
            
        Lamf = boost(P4f) 
        Lammf = boost(P4fm)  
        P3if = np.dot(Lamf,P4i)[1:4]
        P3ifmag = np.sqrt(np.sum([P3if[ii]**2 for ii in range(3)]))
        hatP3if = P3if/P3ifmag
#         print(hatP3if)
        
        imagpart = 0
         
        if 1 + zzf > 0:
            imagpart += 1
        if zzf - 1 > 0:
            imagpart -= 1
        
        nonregQ = np.log(np.abs(lognumf/logdenomf))/np.pi + I * imagpart 
        
        temp = Lammf[nu,0] * omegaqaCM * nonregQ
#         print(nonregQ)

        boostspat = np.sum([Lammf[nu,ii+1] * hatP3if[ii] for ii in range(3)])
        spat = boostspat* qCM * ( zzf * nonregQ - 2/np.pi)
        
        if tempspatorboth=='temp':
            Disc+= I/denom * temp#(temp + spat)
        elif tempspatorboth=='spat':
            Disc+= I/denom * spat#(temp + spat)
        else:
            Disc+= I/denom * (temp + spat) 
        
    if Eist >= (ma+mb) and (fiorboth != 'f'):
        
        zzi = zz(P4i,P4f,ma,mb)
        lognumi = (1 + zzi) 
        logdenomi = (1 - zzi)
        if lognumi*logdenomi == 0:
            return float('nan')
        
        qCM2 = qq2(P4i, ma, mb)
        omegaqaCM = np.sqrt(qCM2 + ma**2)
        # Get the correct branch cut
        if qCM2 < 0:
            qCM = I * np.sqrt(-qCM2)
        else:
            qCM = np.sqrt(qCM2)
            
        Lami = boost(P4i) 
        Lammi = boost(P4im)  
        P3fi = np.dot(Lami,P4f)[1:4]
        P3fimag = np.sqrt(np.sum([P3fi[ii]**2 for ii in range(3)]))
        hatP3fi = P3fi/P3fimag
#         print(hatP3fi)
        
        imagpart = 0
         
        # Conjugate the second leg of one of the diagrams  
        if fiorboth == 'both':
            if 1 + zzi > 0:
                imagpart -= 1
            if zzi - 1 > 0:
                imagpart += 1
        else:
            if 1 + zzi > 0:
                imagpart += 1
            if zzi - 1 > 0:
                imagpart -= 1
        
#         print(imagpart)
        nonregQ = np.log(np.abs(lognumi/logdenomi))/np.pi + I * imagpart 

        temp = Lammi[nu,0] * omegaqaCM * nonregQ
        
        boostspat = np.sum([Lammi[nu,ii+1] * hatP3fi[ii] for ii in range(3)])
        spat = boostspat * qCM * ( zzi * nonregQ - 2/np.pi)
        
#         print(qCM , zzi , nonregQ)
#         print(spat)
#         print(temp+spat)
#         print(denom)
        
            
        if tempspatorboth=='temp':
            Disc+= I/denom * temp#(temp + spat)
        elif tempspatorboth=='spat':
            Disc+= I/denom * spat#(temp + spat)
        else:
            Disc+= I/denom * (temp + spat) 
        
    return Disc


# In[58]:


PFsq = ((2 * np.pi)/6.)**2
PIsq = 0* ((2 * np.pi)/6.)**2

EI = [2.05]

EF = np.linspace(1.9,4,num=50)
print(len(EF))
IA0 = np.zeros(len(EF))
IA0i = np.zeros(len(EF))
IA0disc = np.zeros(len(EF))
IA0disci = np.zeros(len(EF))
IA0discb = np.zeros(len(EF))
IA0discbi = np.zeros(len(EF))

part = 'temp'

for ei in EI:
    for nn, ef in enumerate(EF):
        
        if nn == 30:
            print(0, ef, [0,0,np.sqrt(PFsq)], ei, [0,0,-np.sqrt(PIsq)], m1, m2, 'both')
        
        IA0disctot = DiscIAnu_PineqPf_play(0, ef, [0,0,np.sqrt(PFsq)], ei, [0,0,-np.sqrt(PIsq)], m1, m2, 'i',part)
        
        IA0[nn] = np.real(IA0disctot)
        IA0i[nn] = np.imag(IA0disctot)
        
        IA0disctot = DiscIAnu_PineqPf_play(0, ef, [0,0,np.sqrt(PFsq)], ei, [0,0,-np.sqrt(PIsq)], m1, m2, 'f',part)
        
        IA0disc[nn] = np.real(IA0disctot)
        IA0disci[nn] = np.imag(IA0disctot)
        
        IA0disctot = DiscIAnu_PineqPf_play(0, ef, [0,0,np.sqrt(PFsq)], ei, [0,0,-np.sqrt(PIsq)], m1, m2, 'both', part)
        
        IA0discb[nn] = np.real(IA0disctot)
        IA0discbi[nn] = np.imag(IA0disctot)
        
    plt.figure(1,figsize=(10, 6))
    
    plt.axhline(0,ls='--',color='k')
    
    plt.plot(EF,IA0disc, label="Discf, " + r'$E_i^\star=$'+str(ei))
    
    plt.plot(EF,IA0, label="Disci, " +r'$E_i^\star=$'+str(ei))
    
    plt.plot(EF,IA0discb, label="Disc, " +r'$E_i^\star=$'+str(ei))

    plt.figure(2,figsize=(10, 6))
    
    plt.plot(EF,IA0disci, label="Discf, " + r'$E_i^\star=$'+ str(ei))
    
    plt.plot(EF,IA0i, label="Disci, " + r'$E_i^\star=$'+ str(ei))
    
    plt.plot(EF,IA0discbi, label="Disc, " +r'$E_i^\star=$'+str(ei))
    
plt.figure(1)
plt.title(r'$I_{A0}$'+" Real part")
plt.legend()


plt.figure(2)
plt.title(r'$I_{A0}$'+" Imag part")
plt.legend()


# ## Definitions and quick check of the Feynman parameter method

# In[7]:


PFsq = ((2 * np.pi)/6.)**2
ei = 2.05
ef = 2.05
m1 = 1
m2 = 1

q2 = (np.sqrt(ef**2 + PFsq) - ei)**2 - PFsq
si = ei**2
sf = ef**2

def realxF1(x,ls,ks):
    return np.real(x*F1(x,ls,ks))

def imagxF1(x,ls,ks):
    return np.imag(x*F1(x,ls,ks))

def realF2(x,ls,ks):
    return np.real(F2(x,ls,ks))

def imagF2(x,ls,ks):
    return np.imag(F2(x,ls,ks))

AA2pB_0 = (-4*m2**2)/si + (-m1**2 + m2**2 + si)**2/si**2 

AA2pB_1 = ((-4*(m1**2 - m2**2 - sf))/si + (2*(q2 - sf - si)*(-m1**2 + m2**2 + si))/si**2)

AA2pB_2 = ((q2 - sf - si)**2/si**2 - (4*sf)/si)

xxs = np.roots([AA2pB_2,AA2pB_1,AA2pB_0])

if len(xxs) == 0:
    avoid = (0.5,)
else:    
    if np.angle(xxs[0])==0 or np.angle(xxs[0])==np.pi:
        avoid = xxs
    else:
        avoid = (0.5,)

I11 = integrate.quad(realxF1, 0, 1, args=([m1,m2],[q2,si,sf]), points = avoid)[0]

I11i = integrate.quad(imagxF1, 0, 1, args=([m1,m2],[q2,si,sf]), points = avoid)[0]

I12 = integrate.quad(realF2, 0, 1, args=([m1,m2],[q2,si,sf]), points = avoid)[0]

I12i = integrate.quad(imagF2, 0, 1, args=([m1,m2],[q2,si,sf]), points = avoid)[0]

Pi0 = ei

Pf0 = np.sqrt(ef**2 + PFsq)

print(I11, I11i, I12, I12i)

print([Pf0 * I11 + Pi0 * I12,0,0,np.sqrt(PFsq) * I11])
print([Pf0 * I11i + Pi0 * I12i,0,0,np.sqrt(PFsq) * I11i])

# Keegan results: [(0.030268739364939352+0.030337075324530886j), 0j, 0j, (0.007283427925355396+0.007299871295503539j)]


# # The moment of truth Cutkosky against Feynman

# In[8]:


PFsq = ((2 * np.pi)/6.)**2
PIsq = 0*((2 * np.pi)/6.)**2

PFdotPI = 0

Pdifsq = PIsq + PFsq - 2*PFdotPI

EI = [2.05]

EF = np.linspace(1.9,4,num=10)
print(len(EF))
IA0 = np.zeros(len(EF))
IA0i = np.zeros(len(EF))
IA0disc = np.zeros(len(EF))
IA0disci = np.zeros(len(EF))

ratio = np.zeros(len(EF))


for ei in EI:
    for nn, ef in enumerate(EF):

        q2 = (np.sqrt(ef**2 + PFsq) - np.sqrt(ei**2 + PIsq))**2 - Pdifsq
        si = ei**2
        sf = ef**2

        AA2pB_0 = (-4*m2**2)/si + (-m1**2 + m2**2 + si)**2/si**2 

        AA2pB_1 = ((-4*(m1**2 - m2**2 - sf))/si + (2*(q2 - sf - si)*(-m1**2 + m2**2 + si))/si**2)

        AA2pB_2 = ((q2 - sf - si)**2/si**2 - (4*sf)/si)

        xxs = np.roots([AA2pB_2,AA2pB_1,AA2pB_0])

        if len(xxs) == 0:
            avoid = (0.5,)
        else:    
            if np.angle(xxs[0])==0 or np.angle(xxs[0])==np.pi:
                avoid = xxs
            else:
                avoid = (0.5,)

        I11 = integrate.quad(realxF1, 0, 1, args=([m1,m2],[q2,si,sf]), points = avoid)[0]

        I11i = integrate.quad(imagxF1, 0, 1, args=([m1,m2],[q2,si,sf]), points = avoid)[0]
        
        I12 = integrate.quad(realF2, 0, 1, args=([m1,m2],[q2,si,sf]), points = avoid)[0]

        I12i = integrate.quad(imagF2, 0, 1, args=([m1,m2],[q2,si,sf]), points = avoid)[0]
        
        Pi0 = ei
        
        Pf0 = np.sqrt(ef**2 + PFsq)
        
        IA0[nn] = Pf0 * I11 + Pi0 * I12
        IA0i[nn] = Pf0 * I11i + Pi0 * I12i
        
        IA0disctot = DiscIAnu_PineqPf(0, ef, [0,0,np.sqrt(PFsq)], ei, [0,0,np.sqrt(PIsq)], m1, m2, 'both')
        
        IA0disc[nn] = np.real(IA0disctot)
        IA0disci[nn] = np.imag(IA0disctot)
        
        ratio[nn] = IA0disci[nn]/2./IA0i[nn]
        
    plt.figure(1,figsize=(10, 6))
    
    plt.plot(EF,IA0disc/2, label="Disc/2," + r'$E_i^\star=$'+str(ei))
    
    plt.plot(EF,IA0,'o', label=r'$E_i^\star=$'+str(ei))

    plt.figure(2,figsize=(10, 6))
    
    plt.plot(EF,IA0disci/2, label="Disc/2," + r'$E_i^\star=$'+ str(ei))
    
    plt.plot(EF,IA0i,'o', label=r'$E_i^\star=$'+ str(ei))
    
    plt.figure(3,figsize=(10, 6))
    
    plt.plot(EF,ratio, label="ratio" + r'$E_i^\star=$'+ str(ei))
    
plt.figure(1)
plt.title(r'$I^0_{A}$'+" Real part")
plt.legend()


plt.figure(2)
plt.title(r'$I^0_{A}$'+" Imag part")
plt.legend()

# plt.savefig('../F(1)_integration and triangle/figures/IADisc_nolegend.pdf',
#            bbox_inches='tight', 
#             transparent=True)

plt.figure(3)
plt.title(r'$I^0_{A}$'+" Imag part")
plt.legend()


# In[80]:


PFsq = 0.001

EI = [2.2]

EF = np.linspace(1.9,2.6,num=500)
print(len(EF))
IA0 = np.zeros(len(EF))
IA0i = np.zeros(len(EF))
IA0disc = np.zeros(len(EF))
IA0disci = np.zeros(len(EF))


for ei in EI:
    for nn, ef in enumerate(EF):

        q2 = (np.sqrt(ef**2 + PFsq) - ei)**2 - PFsq
        si = ei**2
        sf = ef**2

        AA2pB_0 = (-4*m2**2)/si + (-m1**2 + m2**2 + si)**2/si**2 

        AA2pB_1 = ((-4*(m1**2 - m2**2 - sf))/si + (2*(q2 - sf - si)*(-m1**2 + m2**2 + si))/si**2)

        AA2pB_2 = ((q2 - sf - si)**2/si**2 - (4*sf)/si)

        xxs = np.roots([AA2pB_2,AA2pB_1,AA2pB_0])

        if len(xxs) == 0:
            avoid = (0.5,)
        else:    
            if np.angle(xxs[0])==0 or np.angle(xxs[0])==np.pi:
                avoid = xxs
            else:
                avoid = (0.5,)

        I11 = integrate.quad(realxF1, 0, 1, args=([m1,m2],[q2,si,sf]), points = avoid)[0]

        I11i = integrate.quad(imagxF1, 0, 1, args=([m1,m2],[q2,si,sf]), points = avoid)[0]
        
        I12 = integrate.quad(realF2, 0, 1, args=([m1,m2],[q2,si,sf]), points = avoid)[0]

        I12i = integrate.quad(imagF2, 0, 1, args=([m1,m2],[q2,si,sf]), points = avoid)[0]
        
        Pi0 = ei
        
        Pf0 = np.sqrt(ef**2 + PFsq)
        
        IA0[nn] = Pf0 * I11 + Pi0 * I12
        IA0i[nn] = Pf0 * I11i + Pi0 * I12i
        
        IA0disctot = DiscIAnu_PineqPf(0, ef, [0,0,np.sqrt(PFsq)], ei, [0,0,0], m1, m2, 'both')
        
        IA0disc[nn] = np.real(IA0disctot)
        IA0disci[nn] = np.imag(IA0disctot)
        
    plt.figure(1,figsize=(10, 6))
    
    plt.plot(EF,IA0disc/2, label="Disc/2," + r'$E_i^\star=$'+str(ei))
    
    plt.plot(EF,IA0,'o', label=r'$E_i^\star=$'+str(ei))

    plt.figure(2,figsize=(10, 6))
    
    plt.plot(EF,IA0disci/2, label="Disc/2," + r'$E_i^\star=$'+ str(ei))
    
    plt.plot(EF,IA0i,'o', label=r'$E_i^\star=$'+ str(ei))
    
plt.figure(1)
plt.title(r'$I_{A0}$'+" Real part")
plt.legend()


plt.figure(2)
plt.title(r'$I_{A0}$'+" Imag part")
plt.legend()


# In[10]:


PFsq = ((2 * np.pi)/6.)**2
PIsq = 0*((2 * np.pi)/6.)**2

PFdotPI = 0

Pdifsq = PIsq + PFsq - 2*PFdotPI

EI = [2.05]

EF = np.linspace(1.9,4,num=10)
print(len(EF))
IA3 = np.zeros(len(EF))
IA3i = np.zeros(len(EF))
IA3disc = np.zeros(len(EF))
IA3disci = np.zeros(len(EF))

ratio = np.zeros(len(EF))


for ei in EI:
    for nn, ef in enumerate(EF):

        q2 = (np.sqrt(ef**2 + PFsq) - np.sqrt(ei**2 + PIsq))**2 - Pdifsq
        si = ei**2
        sf = ef**2

        AA2pB_0 = (-4*m2**2)/si + (-m1**2 + m2**2 + si)**2/si**2 

        AA2pB_1 = ((-4*(m1**2 - m2**2 - sf))/si + (2*(q2 - sf - si)*(-m1**2 + m2**2 + si))/si**2)

        AA2pB_2 = ((q2 - sf - si)**2/si**2 - (4*sf)/si)

        xxs = np.roots([AA2pB_2,AA2pB_1,AA2pB_0])

        if len(xxs) == 0:
            avoid = (0.5,)
        else:    
            if np.angle(xxs[0])==0 or np.angle(xxs[0])==np.pi:
                avoid = xxs
            else:
                avoid = (0.5,)

        I11 = integrate.quad(realxF1, 0, 1, args=([m1,m2],[q2,si,sf]), points = avoid)[0]

        I11i = integrate.quad(imagxF1, 0, 1, args=([m1,m2],[q2,si,sf]), points = avoid)[0]
        
        I12 = integrate.quad(realF2, 0, 1, args=([m1,m2],[q2,si,sf]), points = avoid)[0]

        I12i = integrate.quad(imagF2, 0, 1, args=([m1,m2],[q2,si,sf]), points = avoid)[0]
        
        Pi3 = np.sqrt(PIsq)
        
        Pf3 = np.sqrt(PFsq)
        
        print('Warning!\nMake sure the direction of Pi3 and Pf3 make sense!!!')
        print('Pi3=',Pi3,', Pf3=',Pf3)
        print('\n--')
        
        IA3[nn] = Pf3 * I11 + Pi3 * I12
        IA3i[nn] = Pf3 * I11i + Pi3 * I12i
        
        IA3disctot = DiscIAnu_PineqPf(3, ef, [0,0,np.sqrt(PFsq)], ei, [0,0,np.sqrt(PIsq)], m1, m2, 'both')
        
        IA3disc[nn] = np.real(IA3disctot)
        IA3disci[nn] = np.imag(IA3disctot)
        
        ratio[nn] = IA3disci[nn]/2./IA3i[nn]
        
    plt.figure(1,figsize=(10, 6))
    
    plt.plot(EF,IA3disc/2, label="Disc/2," + r'$E_i^\star=$'+str(ei))
    
    plt.plot(EF,IA3,'o', label=r'$E_i^\star=$'+str(ei))

    plt.figure(2,figsize=(10, 6))
    
    plt.plot(EF,IA3disci/2, label="Disc/2," + r'$E_i^\star=$'+ str(ei))
    
    plt.plot(EF,IA3i,'o', label=r'$E_i^\star=$'+ str(ei))
    
    plt.figure(3,figsize=(10, 6))
    
    plt.plot(EF,ratio, label="ratio" + r'$E_i^\star=$'+ str(ei))
    
plt.figure(1)
plt.title(r'$I^3_{A}$'+" Real part")
plt.legend()


plt.figure(2)
plt.title(r'$I^3_{A}$'+" Imag part")
plt.legend()


plt.figure(3)
plt.title(r'$I^0_{A}$'+" Imag part")
plt.legend()


# # Nice figures

# In[11]:


PFsq = ((2 * np.pi)/6.)**2
PIsq = 0*((2 * np.pi)/6.)**2

PFdotPI = 0

Pdifsq = PIsq + PFsq - 2*PFdotPI

ei = 2.05

EF = np.linspace(1.9,4,num=500)
print(len(EF))
IA0 = np.zeros(len(EF))
IA0i = np.zeros(len(EF))
IA0disc = np.zeros(len(EF))
IA0disci = np.zeros(len(EF))

IA3 = np.zeros(len(EF))
IA3i = np.zeros(len(EF))
IA3disc = np.zeros(len(EF))
IA3disci = np.zeros(len(EF))


for nn, ef in enumerate(EF):

    q2 = (np.sqrt(ef**2 + PFsq) - np.sqrt(ei**2 + PIsq))**2 - Pdifsq
    si = ei**2
    sf = ef**2

    AA2pB_0 = (-4*m2**2)/si + (-m1**2 + m2**2 + si)**2/si**2 

    AA2pB_1 = ((-4*(m1**2 - m2**2 - sf))/si + (2*(q2 - sf - si)*(-m1**2 + m2**2 + si))/si**2)

    AA2pB_2 = ((q2 - sf - si)**2/si**2 - (4*sf)/si)

    xxs = np.roots([AA2pB_2,AA2pB_1,AA2pB_0])

    if len(xxs) == 0:
        avoid = (0.5,)
    else:    
        if np.angle(xxs[0])==0 or np.angle(xxs[0])==np.pi:
            avoid = xxs
        else:
            avoid = (0.5,)

    I11 = integrate.quad(realxF1, 0, 1, args=([m1,m2],[q2,si,sf]), points = avoid)[0]

    I11i = integrate.quad(imagxF1, 0, 1, args=([m1,m2],[q2,si,sf]), points = avoid)[0]

    I12 = integrate.quad(realF2, 0, 1, args=([m1,m2],[q2,si,sf]), points = avoid)[0]

    I12i = integrate.quad(imagF2, 0, 1, args=([m1,m2],[q2,si,sf]), points = avoid)[0]

    Pi0 = ei

    Pf0 = np.sqrt(ef**2 + PFsq)

    IA0[nn] = Pf0 * I11 + Pi0 * I12
    IA0i[nn] = Pf0 * I11i + Pi0 * I12i

    IA0disctot = DiscIAnu_PineqPf(0, ef, [0,0,np.sqrt(PFsq)], ei, [0,0,np.sqrt(PIsq)], m1, m2, 'both')

    IA0disc[nn] = np.real(IA0disctot)
    IA0disci[nn] = np.imag(IA0disctot)

    Pi3 = np.sqrt(PIsq)

    Pf3 = np.sqrt(PFsq)

    print('Warning!\nMake sure the direction of Pi3 and Pf3 make sense!!!')
    print('Pi3=',Pi3,', Pf3=',Pf3)
    print('\n--')

    IA3[nn] = Pf3 * I11 + Pi3 * I12
    IA3i[nn] = Pf3 * I11i + Pi3 * I12i

    IA3disctot = DiscIAnu_PineqPf(3, ef, [0,0,np.sqrt(PFsq)], ei, [0,0,np.sqrt(PIsq)], m1, m2, 'both')

    IA3disc[nn] = np.real(IA3disctot)
    IA3disci[nn] = np.imag(IA3disctot)
    
print(IA0i[-10:])


# In[16]:


# Get ready to plot

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.rc('font', family='serif')
plt.rc('font', size=14)
plt.rc('axes.formatter', useoffset = False)
plt.rc('lines', linewidth=2)

# Colors
JLab_red = (192./255, 39./255, 45./255)
JLab_orange = (249./255, 102./255, 0./255)
JLab_blue = (47./255, 122./255, 121./255)
JLab_green = (65./255, 125./255, 10./255)


# In[33]:


plt.figure(1,figsize=(10, 3))

plt.plot(EF,IA0,'o', label='Feyn params', color=JLab_orange)

plt.plot(EF,IA0disc/2, label="Disc/2", color=JLab_red)

plt.xlim([EF[0],EF[-1]])

plt.xlabel(r'$E^\star_f/m$')

plt.figure(2,figsize=(10, 3))

plt.plot(EF,IA0i,'o', label='Feyn params', color=JLab_orange)

plt.plot(EF,IA0disci/2, label="Disc/2", color=JLab_red)

plt.xlim([EF[0],EF[-1]])

plt.xlabel(r'$E^\star_f/m$')
    
plt.figure(1)
plt.title(r'$mI^0_{A}$'+" Real part, " + r'$E_i^\star='+ str(ei) + r'm$, $\vec{P_i}=0$, $\vec{P_f} = [001]2\pi/L$, $L=6$')
plt.legend()

# plt.savefig('../Derivations with self notation/figures/IA0Re.pdf',
#            bbox_inches='tight', 
#             transparent=True)

plt.figure(2)
plt.title(r'$mI^0_{A}$'+" Imag part, " + r'$E_i^\star='+ str(ei) + r'm$, $\vec{P_i}=0$, $\vec{P_f} = [001]2\pi/L$, $L=6$')
plt.legend()

# plt.savefig('../Derivations with self notation/figures/IA0Im.pdf',
#            bbox_inches='tight', 
#             transparent=True)

plt.figure(3,figsize=(10, 3))

plt.plot(EF,IA3,'o', label='Feyn params', color=JLab_orange)

plt.plot(EF,IA3disc/2, label="Disc/2", color=JLab_red)

plt.xlim([EF[0],EF[-1]])

plt.xlabel(r'$E^\star_f/m$')

plt.figure(4,figsize=(10, 3))

plt.plot(EF,IA3i,'o', label='Feyn params', color=JLab_orange)

plt.plot(EF,IA3disci/2, label="Disc/2", color=JLab_red)

plt.xlim([EF[0],EF[-1]])

plt.xlabel(r'$E^\star_f/m$')

    
plt.figure(3)
plt.title(r'$mI^3_{A}$'+" Real part, " + r'$E_i^\star='+ str(ei) + r'm$, $\vec{P_i}=0$, $\vec{P_f} = [001]2\pi/L$, $L=6$')
plt.legend()

# plt.savefig('../Derivations with self notation/figures/IA3Re.pdf',
#            bbox_inches='tight', 
#             transparent=True)


plt.figure(4)
plt.title(r'$mI^3_{A}$'+" Imag part, " + r'$E_i^\star='+ str(ei) + r'm$, $\vec{P_i}=0$, $\vec{P_f} = [001]2\pi/L$, $L=6$')
plt.legend()

# plt.savefig('../Derivations with self notation/figures/IA3Im.pdf',
#            bbox_inches='tight', 
#             transparent=True)


# In[ ]:




