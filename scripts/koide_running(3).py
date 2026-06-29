#!/usr/bin/env python3
"""SM 1-loop running of K(mu). Seed-free (deterministic ODE)."""
import numpy as np
from scipy.integrate import solve_ivp
def K_of(m): m=np.asarray(m,float); return m.sum()/np.sqrt(m).sum()**2
v=246.0; MZ=91.1876; k=1/(16*np.pi**2)
msbar=[0.486570,102.7181,1746.24]                 # MS-bar lepton masses at M_Z (MeV)
yl0=[np.sqrt(2)*m*1e-3/v for m in msbar]; yt0=0.95
g1=np.sqrt(4*np.pi*(5/3)/127.951/(1-0.23122)); g2=np.sqrt(4*np.pi/127.951/0.23122); g3=np.sqrt(4*np.pi*0.1179)
def rhs(t,y):
    g1,g2,g3,yt,*yl=y; T=3*yt**2
    # Lepton-Yukawa gauge term, GUT-normalized g1 (g1^2 = 5/3 g'^2):
    # non-GUT form 9/4 g'^2 + 9/4 g2^2  ->  (9/4)(3/5) g1^2 + 9/4 g2^2 = 27/20 g1^2 + 9/4 g2^2.
    # (Consistent with b1=41/10 and the top term 17/20 g1^2 below. This term is
    #  flavour-universal and cancels in K, so K is insensitive to its exact value.)
    gl=(27/20)*g1**2+(9/4)*g2**2
    dyl=[k*l*((3/2)*l**2+T-gl) for l in yl]
    dyt=k*yt*((9/2)*yt**2+T-((17/20)*g1**2+(9/4)*g2**2+8*g3**2))
    return [(41/10)*g1**3*k,(-19/6)*g2**3*k,(-7)*g3**3*k,dyt,*dyl]
sol=solve_ivp(rhs,[0,np.log(1e16/MZ)],[g1,g2,g3,yt0,*yl0],dense_output=True,rtol=1e-11,atol=1e-13)
print(f"{'mu (GeV)':>10} | {'K(mu)':>11} | {'K-2/3':>11} | {'dK/dlnmu':>11}")
t_max=np.log(1e16/MZ); h=1e-4
for mu in [MZ,1e3,1e9,1e16]:
    t=np.log(mu/MZ); m=[yi*v/np.sqrt(2) for yi in sol.sol(t)[4:7]]
    if t+h>t_max:                       # backward difference at the endpoint
        mm=[yi*v/np.sqrt(2) for yi in sol.sol(t-h)[4:7]]; dK=(K_of(m)-K_of(mm))/h
    else:
        mp=[yi*v/np.sqrt(2) for yi in sol.sol(t+h)[4:7]]; dK=(K_of(mp)-K_of(m))/h
    print(f"{mu:10.0e} | {K_of(m):.8f} | {K_of(m)-2/3:+.2e} | {dK:+.2e}")
