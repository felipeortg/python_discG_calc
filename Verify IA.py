#!/usr/bin/env python
# coding: utf-8

# In[8]:


import IAfunctions as IAfuns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


# In[12]:


plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.rc('font', family='serif')


# In[15]:


m1 = 1
m2 = 1

PFsq = (2*np.pi/6)**2

EI = [1.999,2-1e-5,2+1e-5,2.01,2.05,2.2,2.26]
labs =['1.99',r'$2-10^{-5}$',r'$2+10^{-5}$','2.01','2.05','2.2','2.26']
# EI=[2.3]
EF = np.r_[1.9:2.6:5000j]


for nn, ei in enumerate(EI):

    plt.figure(nn, figsize=(5.25,3.75))


    ax1 = plt.subplot(211)
    ax1.axhline(0,color='k',linewidth=1)

    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')

    ax2 = plt.subplot(212, sharex=ax1)
    ax2.axhline(0,color='k',linewidth=1)

    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.yaxis.set_ticks_position('left')
    ax2.xaxis.set_ticks_position('bottom')


    if len(EI)==1:
        col = cm.plasma(0)
    else:         
    #     Color by value
    #     col = cm.plasma((ei-EI[0])/(EI[-1]-EI[0]))
    #     Color by element
        col = cm.plasma(2.5/3*(nn)/(len(EI)-1))
    
    IA = np.array([
        0.5*IAfuns.DiscIA_Riem(ef, [0,0,np.sqrt(PFsq)], ei, [0,0,0], m1, m2, 'both')
                   for ef in EF])

    ax1.plot(EF,np.real(IA), color=col, label = labs[nn])
    
    ax2.plot(EF,np.imag(IA), color=col)

    ax1.set_xlim([EF[0],EF[-1]])
    ax1.set_ylabel(r'$m^2$Real Sing $\mathcal G$')
    ax1.legend(fontsize=9, title=r'$E_i^\star/m$', ncol = 2)

    ax2.set_xlabel(r'$E_f^\star/m$')
    ax2.set_ylabel(r'$m^2$Imag Sing $\mathcal G$')
    ax2.set_ylim([0,.076])

plt.show()
# In[ ]:




