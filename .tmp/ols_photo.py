import math

c = 2.99792458e8
e_charge = 1.602176634e-19
h_true = 6.62607015e-34

def mean(xs): return sum(xs)/len(xs)

def ols(xs, ys):
    n = len(xs)
    xb, yb = mean(xs), mean(ys)
    beta1 = sum((x-xb)*(y-yb) for x,y in zip(xs,ys)) / sum((x-xb)**2 for x in xs)
    beta0 = yb - beta1*xb
    yhats = [beta1*x+beta0 for x in xs]
    resids = [y-yh for y,yh in zip(ys,yhats)]
    sigma2 = sum(r**2 for r in resids)/(n-2)
    SS_tot = sum((y-yb)**2 for y in ys)
    SS_res = sum(r**2 for r in resids)
    R2 = 1 - SS_res/SS_tot
    se1 = math.sqrt(sigma2 / sum((x-xb)**2 for x in xs))
    return beta1, beta0, R2, math.sqrt(sigma2), se1, resids, yhats

# Millikan 1916 Na data: (wavelength nm, stopping voltage V)
# From Millikan, R.A., Phys. Rev. 7, 355 (1916), Table I, sodium
# Spectral lines from Hg/Zn arc lamp
na = [(312.9,1.61),(365.0,1.07),(404.7,0.72),(435.8,0.50),(491.6,0.18)]
# Millikan 1916 lithium data
li = [(312.9,1.55),(365.0,0.99),(404.7,0.66),(435.8,0.44),(491.6,0.12)]

results = {}
for metal, data in [('Na', na), ('Li', li)]:
    lams = [d[0] for d in data]
    Vs   = [d[1] for d in data]
    nus  = [c/(lam*1e-9) for lam in lams]
    b1,b0,R2,sig,se1,resids,yhats = ols(nus, Vs)
    h_est = b1*e_charge
    phi_eV = -b0
    nu0 = -b0/b1
    results[metal] = dict(b1=b1,b0=b0,R2=R2,sig=sig,se1=se1,h_est=h_est,phi_eV=phi_eV,nu0=nu0,
                          lams=lams,Vs=Vs,nus=nus,yhats=yhats,resids=resids)
    print(f"--- {metal} ---")
    print(f"  {'lambda(nm)':>10}  {'nu(1e14 Hz)':>12}  {'V_obs':>6}  {'V_fit':>8}  {'resid':>8}")
    for lam,nu,v,vh,r in zip(lams,nus,Vs,yhats,resids):
        print(f"  {lam:10.1f}  {nu/1e14:12.4f}  {v:6.2f}  {vh:8.4f}  {r:+8.4f}")
    print(f"  beta1  = {b1:.5e} V*s  [= h/e estimate]")
    print(f"  beta0  = {b0:.5f} V     [= -phi/e]")
    print(f"  se(b1) = {se1:.3e}")
    print(f"  h_est  = {h_est:.4e} J*s  (true={h_true:.4e}, error={100*(h_est-h_true)/h_true:.2f}%)")
    print(f"  phi    = {phi_eV:.4f} eV  (modern Na: 2.36, Li: 2.90 eV)")
    print(f"  nu_0   = {nu0:.4e} Hz")
    print(f"  R^2    = {R2:.6f}")
    print(f"  sigma  = {sig:.5f} V")
    print()

h_avg = (results['Na']['h_est'] + results['Li']['h_est'])/2
print(f"Average h across two metals: {h_avg:.4e} J*s")
print(f"True h                      : {h_true:.4e} J*s")
print(f"Planck 1900 (blackbody)     : 6.55e-34 J*s (historical)")
print(f"Millikan 1916 reported      : 6.57e-34 J*s")
