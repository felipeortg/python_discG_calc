#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-09-03 16:03:12
# @Author  : Felipe G. Ortega-Gama (felipeortegagama@gmail.com)
# @Version : 1.0
# Triangle loop

# Insertion for particle:
# Lam1 for Feynman integral
# mb for Cutkosky discontinuity


import numpy as np
from scipy import integrate
from scipy import optimize
from scipy import special
I = complex(0,1.)

###########
# Define functions for the calculation with Feynman parameters
###########
def y_noie(x, Lam1, Lam2, pm, q2, si, sf):
    
    x = complex(x, 0)
    
    if pm == 'p':
        pm = 1
    else:
        pm = -1 
              
    AA = 1 + (Lam2**2 - Lam1**2 + x * (q2 - sf - si)) / si
    
    BB = -4 * (Lam2**2 - x * (Lam2**2 - Lam1**2) - x * (1 - x) * sf) / si
    
    return 0.5 * (AA + pm * np.sqrt(AA**2 + BB))

# Singularities of the integrand
def avoid_points(m1, m2, si, sf, q2):

    # Divergences due to the division
    AA2pB_0 = (-4*m2**2)/si + (-m1**2 + m2**2 + si)**2/si**2 
    
    AA2pB_1 = ((-4*(m1**2 - m2**2 - sf))/si + (2*(q2 - sf - si)*(-m1**2 + m2**2 + si))/si**2)
    
    AA2pB_2 = ((q2 - sf - si)**2/si**2 - (4*sf)/si)
    
    xxs = np.roots([AA2pB_2,AA2pB_1,AA2pB_0])
    
    avoid = []

    if np.angle(xxs[0])==0 or np.angle(xxs[0])==np.pi:
        
        if np.real(xxs[0]) > 0 and np.real(xxs[0]) < 1:
            avoid.extend([xxs[0]])
        if np.real(xxs[1]) > 0 and np.real(xxs[1]) < 1:
            avoid.extend([xxs[1]])
    
    # Divergences due to the logarithm
    # Evaluate at the borders to find sign changes
    yp0 = y_noie(0, m1, m2, 'p', q2, si, sf)
    yp1 = y_noie(1, m1, m2, 'p', q2, si, sf)

    ym0 = y_noie(0, m1, m2, 'm', q2, si, sf)
    ym1 = y_noie(1, m1, m2, 'm', q2, si, sf)
    
    if yp0.real * yp1.real < 0:
        def realy(x):
            return y_noie(x, m1, m2, 'p', q2, si, sf).real

        avoid.extend([optimize.brentq(realy, 0, 1)])

    if ym0.real * ym1.real < 0:
        def realy(x):
            return y_noie(x, m1, m2, 'm', q2, si, sf).real

        avoid.extend([optimize.brentq(realy, 0, 1)])

    if (1 - yp0.real) * (- yp1.real) < 0:
        def realy(x):
            return 1 - x - y_noie(x, m1, m2, 'p', q2, si, sf).real

        avoid.extend([optimize.brentq(realy, 0, 1)])        

    if (1 - ym0.real) * (- ym1.real) < 0:
        def realy(x):
            return 1 - x - y_noie(x, m1, m2, 'm', q2, si, sf).real

        avoid.extend([optimize.brentq(realy, 0, 1)]) 

            
    if len(avoid) == 0:
        avoid = (0.5,)

    return avoid
    
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


def F1_tt(x, Lams, kins):
    
    x = complex(x, 0)
    
    q2 = kins[0]
    si = kins[1]
    sf = kins[2]
    
    yp = y_noie(x, Lams[0], Lams[1], 'p', q2, si, sf)
    ym = y_noie(x, Lams[0], Lams[1], 'm', q2, si, sf)
    
    
    ffcoef = 1. / (si * (4 * np.pi)**2 * (yp - ym))
  

    return ffcoef * (antiderlog(0, 1 - x, yp, 'p') - antiderlog(0, 1 - x, ym, 'm'))

# Receive value of masses and kinematics 
# BEWARE: 
# this uses the lower case q = Pf - Pi)
# the ls[0] is the particle mass with the insertion
def I00_tt(ls, ks):
      
    q2 = ks[0]
    si = ks[1]
    sf = ks[2]
    
    m1 = ls[0]
    m2 = ls[1]
    
    def realF(x,ls,ks):
        return np.real(F1_tt(x,ls,ks))

    def imagF(x,ls,ks):
        return np.imag(F1_tt(x,ls,ks))
    
    avoid = avoid_points(m1, m2, si, sf, q2)
    
    
    ff = integrate.quad(realF, 0, 1, args=([m1,m2],[q2,si,sf]), points = avoid)[0]
    ffi = integrate.quad(imagF, 0, 1, args=([m1,m2],[q2,si,sf]), points = avoid)[0]
    
    return ff + I * ffi

###########
# Define functions for the calculation Discontinuity
###########

### Kinematic functions
def dot_4vec(quadvec1,quadvec2):
    return (quadvec1[0]*quadvec2[0]) - sum([(quadvec1[ii])*quadvec2[ii] for ii in range(1,4)])

def square_4vec(quadvec):
    return dot_4vec(quadvec,quadvec)

def qq2(P4v, ma, mb):
    ss = square_4vec(P4v) 
    return (ss - 2*(ma**2 + mb**2) + (ma**2 - mb**2)**2/ss)/4

# Boost matrix
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

def beta_3vec(Est, P3):
    P3m2 = np.dot(P3,P3)

    beta2 = P3m2/(Est**2 + P3m2)

    if P3m2 != 0:
        return np.sqrt(beta2/P3m2)* np.array(P3)
    else:
        return np.array([0,0,0])

### The z^star function 
def zz(P4vCM, P4vNOCM, ma, mb):
    qCM2 = qq2(P4vCM, ma, mb)
    
    # Get the correct branch cut
    if qCM2 < 0:
        qCM = I * np.sqrt(-qCM2)
    else:
        qCM = np.sqrt(qCM2)
        
    omegaqaCM = np.sqrt(qCM2 + ma**2)
    omegaqbCM = np.sqrt(qCM2 + mb**2)
    
    Lamb = boost(P4vCM)
    
    P4v = np.dot(Lamb, P4vNOCM)
    
    P3m2 = sum([(P4v[ii])**2 for ii in range(1,4)])
    
    P3m = np.sqrt(P3m2)
    
    return (P3m2 + omegaqbCM**2 - (P4v[0]-omegaqaCM)**2)/(2*P3m*qCM)


# Functions for calculation with Cutkosky
def log_cut(P4vCM, P4vNOCM, ma, mb, cont, conj):

    zst = zz(P4vCM,P4vNOCM,ma,mb)

    Est = np.sqrt(square_4vec(P4vCM))
    
    # Some extra care is needed at threshold
    if Est == (ma+mb):
        lognum = 1 
        logdenom = 1
    else:
        lognum = (1 + zst) 
        logdenom = (1 - zst)

    if Est >= (ma+mb):
        
        imagpart = 0

        if 1 + zst > 0:
            imagpart += 1
        if zst - 1 > 0:
            imagpart -= 1

        if conj:
            imagpart *= -1

        return I * (np.log(np.abs(lognum/logdenom))/np.pi + I * imagpart )
    
    # Analytic continuation below threshold    
    if cont:  
        # The addition of an i*pi factor to select the correct Riem Sheet
        return I * (np.log(lognum/logdenom)+ I*np.pi)/np.pi
    else:
        return 0

# Improve the Riem sheet understanding
def log_cut_Riem(P4vCM, P4vNOCM, ma, mb):
    
    Est = np.sqrt(square_4vec(P4vCM))

    # Some extra care is needed at threshold
    if Est == (ma+mb):
        P4vCM[0] -= 1e-4 # Check to know which side of infinity should be evaluated
        zst = zz(P4vCM,P4vNOCM,ma,mb)
        if np.imag(zst) > 0 :
            lognum = -1 
            logdenom = -1
        else:
            lognum = 1 
            logdenom = 1

        # lognum = 1 
        # logdenom = 1
    else:

        zst = zz(P4vCM,P4vNOCM,ma,mb)

        lognum = zst + 1
        logdenom = zst - 1

    # return I * (np.log(-I*lognum) - np.log(I*logdenom) + I*np.pi)/np.pi  
    return I * (np.log(-I*lognum) - np.log(I*logdenom))/np.pi    


def comoving_cut(Est, EstNO, ma, mb, cont):
    P4 = np.array([Est,0,0,0])
    
    qCM2 = qq2(P4, ma, mb)
    
    # Get the correct branch cut
    if qCM2 < 0:
        qCM = I * np.sqrt(-qCM2)
    else:
        qCM = np.sqrt(qCM2)

    denom = Est * (Est - EstNO) * (EstNO + (mb**2-ma**2)/Est ) 
        
    return I/(4*np.pi) * qCM / denom

def DiscIA(Efst, P3_f, Eist, P3_i, ma, mb, fiorboth, cont, conj):
    betaf = beta_3vec(Efst, P3_f)
    betai = beta_3vec(Eist, P3_i)

    if betaf[0] == betai[0] and betaf[1] == betai[1] and betaf[2] == betai[2]:
        if Efst == Eist:
            if Efst > (ma + mb) or cont:
                return DiscIA_PieqPf(Efst, ma, mb)
        else:
            Disc = 0
            if (fiorboth != 'i'):
                if Efst > (ma + mb) or cont:
                    Disc += comoving_cut(Efst, Eist, ma, mb, cont)
            if (fiorboth != 'f'):
                if Eist > (ma + mb) or cont:
                    Disc += comoving_cut(Eist, Efst, ma, mb, cont)

            return Disc 
    else:
        P3m2f = np.dot(P3_f,P3_f)
        P3m2i = np.dot(P3_i,P3_i)
        P4f = np.array([np.sqrt(Efst**2 + P3m2f),P3_f[0],P3_f[1],P3_f[2]])
        P4i = np.array([np.sqrt(Eist**2 + P3m2i),P3_i[0],P3_i[1],P3_i[2]])
        denom = 16 * np.sqrt(dot_4vec(P4f,P4i)**2 - square_4vec(P4f) * square_4vec(P4i) )

        cuts = 0
        if (fiorboth != 'i'):
            cuts+= log_cut(P4f, P4i, ma, mb, cont, False) # Never conjugate the Final cut
        if (fiorboth != 'f'):
            cuts+= log_cut(P4i, P4f, ma, mb, cont, conj) 

        return cuts/denom

def DiscIA_Riem(Efst, P3_f, Eist, P3_i, ma, mb, fiorboth):
    betaf = beta_3vec(Efst, P3_f)
    betai = beta_3vec(Eist, P3_i)

    if betaf[0] == betai[0] and betaf[1] == betai[1] and betaf[2] == betai[2]:
        if Efst == Eist:
            return DiscIA_PieqPf(Efst, ma, mb)
        else:
            Disc = 0
            if (fiorboth != 'i'):
                Disc += comoving_cut(Efst, Eist, ma, mb, cont)
            if (fiorboth != 'f'):
                Disc += comoving_cut(Eist, Efst, ma, mb, cont)

            return Disc 
    else:
        P3m2f = np.dot(P3_f,P3_f)
        P3m2i = np.dot(P3_i,P3_i)
        P4f = np.array([np.sqrt(Efst**2 + P3m2f),P3_f[0],P3_f[1],P3_f[2]])
        P4i = np.array([np.sqrt(Eist**2 + P3m2i),P3_i[0],P3_i[1],P3_i[2]])
        denom = 16 * np.sqrt(dot_4vec(P4f,P4i)**2 - square_4vec(P4f) * square_4vec(P4i) )

        cuts = 0
        if (fiorboth != 'i'):
            cuts+= log_cut_Riem(P4f, P4i, ma, mb)
        if (fiorboth != 'f'):
            cuts+= log_cut_Riem(P4i, P4f, ma, mb) 

        return cuts/denom


def Tripledelta(Efst, P3_f, Eist, P3_i, ma, mb):
    P3m2f = np.dot(P3_f,P3_f)
    P3m2i = np.dot(P3_i,P3_i)
    
    P4f = np.array([np.sqrt(Efst**2 + P3m2f),P3_f[0],P3_f[1],P3_f[2]])
    
    P4i = np.array([np.sqrt(Eist**2 + P3m2i),P3_i[0],P3_i[1],P3_i[2]])
    
    denom = 8 * np.sqrt(dot_4vec(P4f,P4i)**2 - square_4vec(P4f) * square_4vec(P4i) )
    
    if denom == 0:
        return float('nan')
    
    zzf = zz(P4f,P4i,ma,mb)
    zzi = zz(P4i,P4f,ma,mb)

    heavf = 0
    if 1 + zzf > 0:
        heavf += 1
    if zzf - 1 > 0:
        heavf -= 1

    heavi = 0
    if 1 + zzi > 0:
        heavi += 1
    if zzi - 1 > 0:
        heavi -= 1

    if heavi != heavf:
        print('------')
        print(zzf,zzi)
        print(heavf,heavi)
        print ("There is something that does not make sense at: ", Efst,P3_f, Eist, P3_i)
        
    return -1/denom * heavi


def DiscIA_PineqPf(Efst, P3_f, Eist, P3_i, ma, mb, fiorboth, cont):
    P3m2f = sum([(P3_f[ii])**2 for ii in range(3)])
    P3m2i = sum([(P3_i[ii])**2 for ii in range(3)])
    
    P4f = np.array([np.sqrt(Efst**2 + P3m2f),P3_f[0],P3_f[1],P3_f[2]])
    
    P4i = np.array([np.sqrt(Eist**2 + P3m2i),P3_i[0],P3_i[1],P3_i[2]])
    
    denom = 16 * np.sqrt(dot_4vec(P4f,P4i)**2 - square_4vec(P4f) * square_4vec(P4i) )
    
    if denom == 0:
        return float('nan')
    
    zzf = zz(P4f,P4i,ma,mb)
    zzi = zz(P4i,P4f,ma,mb)
    
    # Some extra care is needed at threshold
    if Efst == (ma+mb): 
        lognumf = 1
        logdenomf = 1
    else:
        lognumf = (1 + zzf) 
        logdenomf = (1 - zzf)
    
    if Eist == (ma+mb):
        lognumi = 1 
        logdenomi = 1
    else:
        lognumi = (1 + zzi) 
        logdenomi = (1 - zzi)
    
    # Return nans at the singularities
    if fiorboth == 'f':
        if lognumf*logdenomf == 0:
            return float('nan')
    elif fiorboth == 'i':
        if lognumi*logdenomi == 0:
            return float('nan')
    else:
        if lognumf*logdenomf*lognumi*logdenomi == 0:
            return float('nan')
         
    
    Disc = 0
 
    if Efst >= (ma+mb) and (fiorboth != 'i'):  

        imagpart = 0 # This part should always cancel for both, leave it as check

        if 1 + zzf > 0:
            imagpart += 1
        if zzf - 1 > 0:
            imagpart -= 1
            
        Disc+= I/denom * (np.log(np.abs(lognumf/logdenomf))/np.pi + I * imagpart )
        
    # Analytic continuation below threshold 
    if Efst < (ma+mb) and (fiorboth != 'i') and cont:  
        # The addition of an i*pi factor to select the correct Riem Sheet
        Disc+= I/denom * (np.log(lognumf/logdenomf) + I*np.pi)/np.pi
        
    if Eist >= (ma+mb) and (fiorboth != 'f'):
        
        imagpart = 0
        
        # Conjugate the second leg of one of the diagrams  
        if fiorboth == 'both':
            if 1 + zzi > 0: # This part should always cancel for both, leave it as check
                imagpart -= 1
            if zzi - 1 > 0:
                imagpart += 1
        else:
            if 1 + zzi > 0:
                imagpart += 1
            if zzi - 1 > 0:
                imagpart -= 1

        Disc+= I/denom * (np.log(np.abs(lognumi/logdenomi))/np.pi + I * imagpart )
    
    # Analytic continuation below threshold    
    if Eist < (ma+mb) and (fiorboth != 'f') and cont:  
        # The addition of an i*pi factor to select the correct Riem Sheet
        Disc+= I/denom * (np.log(lognumi/logdenomi)+ I*np.pi)/np.pi
        
    return Disc

def DiscIA_PiveceqPfvec(Efst, Eist, ma, mb, fiorboth):
    P4f = np.array([Efst,0,0,0])
    
    P4i = np.array([Eist,0,0,0])
    
    qCM2f = qq2(P4f, ma, mb)
    qCM2i = qq2(P4i, ma, mb)
    
    # Get the correct branch cut
    if qCM2i < 0:
        qCMi = I * np.sqrt(-qCM2i)
    else:
        qCMi = np.sqrt(qCM2i)
        
    if qCM2f < 0:
        qCMf = I * np.sqrt(-qCM2f)
    else:
        qCMf = np.sqrt(qCM2f)
    

    Disc = 0
   
    if (fiorboth != 'i'):
        denom = Efst * (Efst - Eist) * (Eist + (mb**2-ma**2)/Efst )
            
        Disc+= qCMf / denom
        
    if (fiorboth != 'f'):
        denom = Eist * (Eist - Efst) * (Efst + (mb**2-ma**2)/Eist )
            
        Disc+= qCMi / denom
        
    return I/(4*np.pi) * Disc


def DiscIA_PieqPf(Est, ma, mb):
    P4 = np.array([Est,0,0,0])
    
    qCM2 = qq2(P4, ma, mb)
    
    # Get the correct branch cut
    if qCM2 < 0:
        qCM = I * np.sqrt(-qCM2)
    else:
        qCM = np.sqrt(qCM2)
        
    return I* (Est**2 - mb**2 + ma**2) /(16*np.pi*qCM*(Est**(1.5))) 

def Step_PineqPf(Efst, P3_f, Eist, P3_i, ma, mb):
    P3m2f = sum([(P3_f[ii])**2 for ii in range(3)])
    P3m2i = sum([(P3_i[ii])**2 for ii in range(3)])
    
    P4f = np.array([np.sqrt(Efst**2 + P3m2f),P3_f[0],P3_f[1],P3_f[2]])
    
    P4i = np.array([np.sqrt(Eist**2 + P3m2i),P3_i[0],P3_i[1],P3_i[2]])
    
    denom = 16 * np.sqrt(dot_4vec(P4f,P4i)**2 - square_4vec(P4f) * square_4vec(P4i) )
    
    if denom == 0:
        return float('nan')
    
    zzf = zz(P4f,P4i,ma,mb)
    zzi = zz(P4i,P4f,ma,mb)
        
    # Added an 'arbitrary' constant, ie 1/denom, to make sure below threshold it also has it
    
    if Efst < ma + mb or Eist < ma + mb :
        return 1/denom
    elif 1 - zzi*zzf > 0:
        return 0
    else:
        return 1/denom
    
def Disc_gen(Efst, P3_f, Eist, P3_i, ma, mb):
    betaf = beta_3vec(Efst, P3_f)
    betai = beta_3vec(Eist, P3_i)


    if betaf[0] == betai[0] and betaf[1] == betai[1] and betaf[2] == betai[2]:
        if Efst == Eist:
            return DiscIA_PieqPf(Efst, ma, mb)
        else:
            return DiscIA_PiveceqPfvec(Efst, Eist, ma, mb, 'both')   
    else:
        return DiscIA_PineqPf(Efst, P3_f, Eist, P3_i, ma, mb, 'both', True)
