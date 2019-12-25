import numpy as np
from itertools import product
from numba import jit,njit,int32,float32

@njit(cache=True)
def dot6(a, b):
    return (a[0]*b[0]+
            a[3]*b[3]+
            a[5]*b[5]+
            2.0*a[1]*b[1]+
            2.0*a[2]*b[2]+
            2.0*a[4]*b[4])

@njit(cache=True)
def dot10(a, b):
    coeff = np.zeros(10, np.float32)
    coeff[0]=1
    coeff[1]=3
    coeff[2]=3
    coeff[3]=1
    coeff[4]=3
    coeff[5]=6
    coeff[6]=3
    coeff[7]=3
    coeff[8]=3
    coeff[9]=1
    coeff*=a*b
    return coeff.sum()

#@njit(cache=True)
def ten6toMat(a):
    ret = np.zeros((3,3), np.float32)
    ret[[0,1,2],[0,1,2]] = a[[0, 3, 5]]
    ret[[0,1],[1,0]] = a[1]
    ret[[0,2],[2,0]] = a[2]
    ret[[1,2],[2,1]] = a[4]
    return ret
    
            
@njit(cache=True)
def randomQuaternion(wiggleRate):
    result = np.zeros(4, dtype=np.float32)

    r1=0
    r2=0
    (y1,y2,y3,y4) = (0,0,0,0)
    while True:
        y1=1.0-2.0*np.random.random()
        y2=1.0-2.0*np.random.random()
        r1=y1**2+y2**2
        if r1<=1:
            break
    
    while True:
        y3=1.0-2.0*np.random.random()
        y4=1.0-2.0*np.random.random()
        r2=y3**2+y4**2
        if r2<=1:
            break

    sr=np.sqrt((1-r1)/r2)

    result[0]=wiggleRate*y1;
    result[1]=wiggleRate*y2;
    result[2]=wiggleRate*y3*sr;
    result[3]=wiggleRate*y4*sr;
    return result

@njit(cache=True)
def randomUniformQuaternion():
    return randomQuaternion(1)

@njit(cache=True)
def initializeRandomQuaternions(xlattice):
    for i in np.ndindex(xlattice.shape[:-1]):
        xlattice[i] = randomUniformQuaternion()

@njit(cache=True)
def getPropertiesFromOrientation(x, parity):
    x11 = x[1] * x[1]
    x22 = x[2] * x[2]
    x33 = x[3] * x[3]
    x01 = x[0] * x[1]
    x02 = x[0] * x[2]
    x03 = x[0] * x[3]
    x12 = x[1] * x[2]
    x13 = x[1] * x[3]
    x23 = x[2] * x[3]

    ex = [2 * (-x22 - x33 + 0.5), 2 * (x12 + x03), 2 * (x13 - x02)]
    ey = [2 * (x12 - x03), 2 * (-x11 - x33 + 0.5), 2 * (x01 + x23)]
    ez = [2 * (x02 + x13), 2 * (-x01 + x23), 2 * (-x22 - x11 + 0.5)]

    xx = np.zeros(6, np.float32)
    yy = np.zeros(6, np.float32)
    zz = np.zeros(6, np.float32)

    xx[0] = ex[0] * ex[0]
    xx[1] = ex[0] * ex[1]
    xx[2] = ex[0] * ex[2]
    xx[3] = ex[1] * ex[1]
    xx[4] = ex[1] * ex[2]
    xx[5] = ex[2] * ex[2]

    yy[0] = ey[0] * ey[0]
    yy[1] = ey[0] * ey[1]
    yy[2] = ey[0] * ey[2]
    yy[3] = ey[1] * ey[1]
    yy[4] = ey[1] * ey[2]
    yy[5] = ey[2] * ey[2]

    zz[0] = ez[0] * ez[0]
    zz[1] = ez[0] * ez[1]
    zz[2] = ez[0] * ez[2]
    zz[3] = ez[1] * ez[1]
    zz[4] = ez[1] * ez[2]
    zz[5] = ez[2] * ez[2]

    I = np.zeros(6, np.float32)
    I[np.array([0,3,5])] = 1
    t20 = np.sqrt(3/2)*(zz - 1/3*I)
    t22 = np.sqrt(1/2)*(xx - yy)

    t32 = np.zeros(10, np.float32)

    #000
    t32[0] = 6.0 * (ex[0] * ey[0] * ez[0]) # 1
    #100
    t32[1] = 2.0 * (ex[0] * ey[0] * ez[1] + # 3
                    ex[0] * ey[1] * ez[0] +
                    ex[1] * ey[0] * ez[0])
    #110
    t32[2] = 2.0 * (ex[0] * ey[1] * ez[1] + # 3
                    ex[1] * ey[0] * ez[1] +
                    ex[1] * ey[1] * ez[0])
    #111
    t32[3] = 6.0 * (ex[1] * ey[1] * ez[1]) # 1
    #200
    t32[4] = 2.0 * (ex[0] * ey[0] * ez[2] + # 3
                    ex[0] * ey[2] * ez[0] +
                    ex[2] * ey[0] * ez[0])
    #210
    t32[5] = (ex[0] * ey[1] * ez[2] +     # 6
              ex[0] * ey[2] * ez[1] +
              ex[1] * ey[0] * ez[2] +
              ex[2] * ey[0] * ez[1] +
              ex[1] * ey[2] * ez[0] +
              ex[2] * ey[1] * ez[0]
             )
    #211
    t32[6] = 2.0 * (ex[1] * ey[1] * ez[2] + # 3
                    ex[1] * ey[2] * ez[1] +
                    ex[2] * ey[1] * ez[1]
                   )
    #220
    t32[7] = 2.0 * (ex[2] * ey[2] * ez[0] + # 3
                    ex[0] * ey[2] * ez[2] +
                    ex[2] * ey[0] * ez[2])
    #221
    t32[8] = 2.0 * (ex[2] * ey[2] * ez[1] + # 3
                    ex[1] * ey[2] * ez[2] +
                    ex[2] * ey[1] * ez[2])
    #222
    t32[9] = 6.0 * (ex[2] * ey[2] * ez[2]) # 1

    t32*=(parity/np.sqrt(6))

    return np.array([ex, ey, ez]), t20, t22, t32

@njit(cache=True)
def getNeighbors(center, lattice):
    ind = np.mod(np.array([
        [ 1,0,0],
        [-1,0,0],
        [0, 1,0],
        [0,-1,0],
        [0,0, 1],
        [0,0,-1]
    ]) + center, lattice.shape[0])
    n = np.zeros((6, lattice.shape[-1]), lattice.dtype)
    n[0] = lattice[ind[0,0], ind[0,1], ind[0,2]]
    n[1] = lattice[ind[1,0], ind[1,1], ind[1,2]]
    n[2] = lattice[ind[2,0], ind[2,1], ind[2,2]]
    n[3] = lattice[ind[3,0], ind[3,1], ind[3,2]]
    n[4] = lattice[ind[4,0], ind[4,1], ind[4,2]]
    n[5] = lattice[ind[5,0], ind[5,1], ind[5,2]]
    return n


@njit(cache=True)
def getEnergy(x, p, nx, npi, lam, tau):
    e, t20, t22, t32 = getPropertiesFromOrientation(x, p)
    Q = t20 + lam*np.sqrt(2)*t22
    energy = 0
    for i in range(nx.shape[0]):
        ei, t20i, t22i, t32i = getPropertiesFromOrientation(nx[i], npi[i])
        Qi = t20i + lam*np.sqrt(2)*t22i
        energy += (-dot6(Q,Qi)-tau*dot10(t32,t32i))/2
    return energy

@njit(cache=True)
def metropolis(dE, temperature):
    if dE < 0:
        return True
    else:
        if np.random.random() < np.exp(-dE/temperature):
            return True
    return False

@njit(cache=True)
def wiggleQuaternion(x, wiggleRate):
    dx = randomQuaternion(wiggleRate)
    x_ = x.copy()
    x_ += dx
    x_ /= np.linalg.norm(x_)
    return x_

@njit(cache=True)
def fluctuation(values):
    fluct = np.zeros_like(values)
    for i in range(fluct.size):
        e = np.random.choice(values, values.size)
        e2 = e*e
        fluct[i] = (e2.mean()-e.mean()**2)
    return fluct.mean()

@jit(forceobj=True,nopython=False,parallel=True)
def doOrientationSweep(lattice, indexes, temperature, lam, tau):
    for _i in indexes:
        particle = lattice[tuple(_i)]
        #nind = getNeighborsIndices(_i, L)
        nx = getNeighbors(_i, lattice['x'])
        npi = getNeighbors(_i, lattice['p'][...,np.newaxis])
        energy1 = getEnergy(particle['x'], particle['p'], nx, npi, lam=lam, tau=tau)
        
        # adjust x
        x_ = wiggleQuaternion(particle['x'], wiggleRate)
        energy2 = getEnergy(x_, particle['p'], nx, npi, lam=lam, tau=tau)
        if metropolis(2*(energy2-energy1), temperature):
            particle['x'] = x_
            particle['energy'] = energy2

        # adjust p
        p_ = -particle['p']
        energy2 = getEnergy(particle['x'], p_, nx, npi, lam=lam, tau=tau)
        if metropolis(2*(energy2-energy1), temperature):
            particle['p'] = p_
            particle['energy'] = energy2

        particle['basis'], particle['t20'], particle['t22'], particle['t32'] = getPropertiesFromOrientation(particle['x'], particle['p'])
    


particle = np.dtype({
    'names':   [ 'index',
                 'x',
                 'basis',
                 't20',
                 't22',
                 't32',
                 'p',
                 'energy',],
    'formats': [ (np.int, (3,)),
                 (np.float32, (4,)),
                 (np.float32, (3,3)),
                 (np.float32, (6,)),
                 (np.float32, (6,)),
                 (np.float32, (10,)),
                 np.int32,
                 np.float32
               ]
})

np.random.seed(42)
L = 9
lattice = np.zeros((L,L,L), dtype=particle)

##random state
# initializeRandomQuaternions(lattice['x'])
# lattice['p'] = np.random.choice([-1,1], L*L*L).reshape(L,L,L)

##ordered state
# lattice['x'] = [1,0,0,0]
# lattice['p'] = 1

##randomised partially ordered state
#<parity>~0.5, "mostly" biaxial nematic
lattice['p'] = np.random.choice([-1,1], lattice.size, p=[0.25,0.75]).reshape(lattice.shape)
lattice['x'] = [1,0,0,0]
for i in np.ndindex(lattice.shape):
    lattice['x'][i] = wiggleQuaternion(lattice['x'][i], 0.02)

lattice['index'] = np.array(list(np.ndindex((L,L,L)))).reshape(L,L,L,3)
#temperature = 0.776
temperature = 1.186
#temperature = 0.5
lam=0.3
tau=1
wiggleRate = 1.1

meanEnergy = np.array([0], np.float32)
energyVariance = np.array([0], np.float32)
wiglleRateValues = np.array([wiggleRate], np.float32)

for it in range(10000):
    indexes = lattice['index'].reshape(-1,3)[np.random.randint(0, lattice.size, lattice.size)]
    doOrientationSweep(lattice, indexes, temperature, lam, tau)

    # compute stats
    meanEnergy = np.append(meanEnergy,lattice['energy'].mean())
    if it > 100:
        energyVariance = np.append(energyVariance,lattice.size*fluctuation(meanEnergy[-100:]))
    wiglleRateValues = np.append(wiglleRateValues, wiggleRate)

    # adjusting of wiggle rate
    if it % 10 == 0 and meanEnergy.size > 101:
        mE = np.array([ m.mean() for m in np.split(meanEnergy[-100:], 4)])
        mR = np.array([ m.mean() for m in np.split(wiglleRateValues[-100:], 4)])
        
        de = np.diff(mE, 1)
        dr = np.diff(mR, 1)
        efirst = de/dr
        etrend = (efirst[-1]-efirst[-2])
        if etrend < 0:
            wiggleRate *= 1.1
        else:
            wiggleRate *= 0.9
    wiggleRate *= 1 + np.random.normal(scale=0.001)
    
    # randomly adjust wiggle rate
    if it % 1000 == 0 and it > 0:
        wiggleRate = np.random.normal(wiggleRate, scale=1.0)

    # print stats
    if it % 50 == 0:
        w,v = np.linalg.eig(lattice['basis'].mean(axis=(0,1,2)))

        mQ = lattice['t20'].mean(axis=(0,1,2)) + lam*np.sqrt(2)*lattice['t22'].mean(axis=(0,1,2))
        mQ = ten6toMat(mQ)
        qw,qv = np.linalg.eig(mQ)

        q0 = np.sum(np.power(lattice['t20'].mean(axis=(0,1,2)),2))
        q2 = np.sum(np.power(lattice['t22'].mean(axis=(0,1,2)),2))
        p = lattice['p'].mean()

        print(f'r={wiggleRate:.3f}, <E>={meanEnergy[-1]:.2f}, var(E)={energyVariance[-1]:.4f}, q0={q0:.6f}, q2={q2:.6f}, p={p:.4f}, qev={qw}')
