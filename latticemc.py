import numpy as np
from itertools import product
from numba import jit,njit,int32,float32

@njit
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

@njit
def randomUniformQuaternion():
    return randomQuaternion(1)

@njit
def initializeRandomQuaternions(orientation):
    for i in np.ndindex(orientation.shape[:-1]):
        orientation[i] = randomUniformQuaternion()

@njit
def getPropertiesFromOrientation(x):
    x11 = x[1] * x[1]
    x22 = x[2] * x[2]
    x33 = x[3] * x[3]
    x01 = x[0] * x[1]
    x02 = x[0] * x[2]
    x03 = x[0] * x[3]
    x12 = x[1] * x[2]
    x13 = x[1] * x[3]
    x23 = x[2] * x[3]

    e = np.zeros((3,3), dtype=np.float32)

    e[:,0] = [2 * (-x22 - x33 + 0.5), 2 * (x12 + x03), 2 * (x13 - x02)]
    e[:,1] = [2 * (x12 - x03), 2 * (-x11 - x33 + 0.5), 2 * (x01 + x23)]
    e[:,2] = [2 * (x02 + x13), 2 * (-x01 + x23), 2 * (-x22 - x11 + 0.5)]
    return e
    # Qx = np.outer(e[:,0], e[:,0])
    # Qy = np.outer(e[:,1], e[:,1])
    # Qz = np.outer(e[:,2], e[:,2])

    # return e, Qx, Qy, Qz

@njit
def getNeighbors(center, orientation):
    ind = np.mod(np.array([
        [ 1,0,0],
        [-1,0,0],
        [0, 1,0],
        [0,-1,0],
        [0,0, 1],
        [0,0,-1]
    ]) + center, orientation.shape[0])
    return np.array([
        list(orientation[ind[0,0], ind[0,1], ind[0,2]]),
        list(orientation[ind[1,0], ind[1,1], ind[1,2]]),
        list(orientation[ind[2,0], ind[2,1], ind[2,2]]),
        list(orientation[ind[3,0], ind[3,1], ind[3,2]]),
        list(orientation[ind[4,0], ind[4,1], ind[4,2]]),
        list(orientation[ind[5,0], ind[5,1], ind[5,2]])
    ], np.float32)


@njit([float32(float32[:],float32[:,:],float32)])
def getEnergy(x, neighbors, t):
    e = getPropertiesFromOrientation(x)
    energy = 0
    for i in range(neighbors.shape[0]):
        en = getPropertiesFromOrientation(neighbors[i])
        energy += np.sum(e*en)
    energy /= -t
    return energy

# particle = np.dtype({
#     'names': ['index','x'],
#     'formats': [ (np.int, (3,)), (np.float32, (4,))]
# })

np.random.seed(42)
L = 9
#orientation = np.zeros((L,L,L), dtype=particle)
orientation = np.zeros((L,L,L,4), dtype=np.float32)
energy = np.zeros((L,L,L), dtype=np.float32)
basis = np.zeros((L,L,L,3,3), dtype=np.float32)
ind = np.array(list(np.ndindex((L,L,L))))
initializeRandomQuaternions(orientation)
temperature = 2
wiggleRate = 0.1

meanEnergy = np.array([0], np.float32)
wiglleRateValues = np.array([wiggleRate], np.float32)

@jit(forceobj=True,nopython=False,parallel=True)
def doOrientationSweep(orientation, indexes):
    global energy, basis, wiggleRate
    for _i in indexes:
        i=tuple(_i)
        x = orientation[i]
        #nind = getNeighborsIndices(_i, L)
        neighbors = getNeighbors(_i, orientation)
        energy1 = getEnergy(x, neighbors, temperature)
        # adjust x
        nx = x.copy()
        dx = randomQuaternion(wiggleRate)
        nx += dx
        nx /= np.linalg.norm(nx)
        # evaluate change
        energy2 = getEnergy(nx, neighbors, temperature)
        if energy2<energy1:
            orientation[i] = nx
            energy[i] = energy2
        else:
            if np.random.random() < np.exp(-2*(energy2-energy1)):
                orientation[i] = nx
                energy[i] = energy2
        basis[i] = getPropertiesFromOrientation(orientation[i])
    


# orientation = np.pad(orientation, 1, mode='wrap')[..., 1:-1]
# ind = ind.reshape(L,L,L,3)

# @jit(forceobj=True,nopython=False,parallel=True)
# def process_subcells(ind, orientation):
#     for x,y,z in product(range(0,3), range(0,3), range(0,3)):
#         indexes = ind[x::3,y::3,z::3,:].reshape(-1,3)
#         doorientationSweep(orientation, indexes)

#from joblib import Parallel, delayed
#def process_subcells_joblib(ind, orientation, parallel):
#    parallel(delayed(doorientationSweep)(orientation, ind[x::3,y::3,z::3,:].reshape(-1,3)) for x,y,z in product(range(0,3), range(0,3), range(0,3)))

#with Parallel(n_jobs=27, prefer='threads') as parallel:

for it in range(10000):
    indexes = ind[np.random.randint(0, ind.shape[0], ind.shape[0])]
    doOrientationSweep(orientation, indexes)
    #process_subcells(ind, orientation)
    #process_subcells_joblib(ind, orientation)
    meanEnergy = np.append(meanEnergy,energy.mean())
    wiglleRateValues = np.append(wiglleRateValues, wiggleRate)
    if meanEnergy.size > 11:
        de = np.diff(meanEnergy, 1)
        dr = np.diff(wiglleRateValues, 1)
        efirst = de/dr
        rfirst = 0.5*(wiglleRateValues[:-1]+ wiglleRateValues[1:])
        d2e = np.diff(efirst, 1)
        dr2 = np.diff(rfirst, 1)
        rsecond = d2e/dr2
        if rsecond[-10].mean() < 0:
            wiggleRate *= 1.01
        else:
            wiggleRate *= 0.99
    wiggleRate *= 1 + np.random.normal(scale=0.01)

    if it % 20 == 0 and it > 0:
        wiggleRate = np.abs(np.random.normal(scale=0.5))
    if it % 50 == 0:
        if it>200:
            de = meanEnergy[-10].mean() - meanEnergy[-200:-180].mean()
            print(de)
        print(wiggleRate)
        w,v = np.linalg.eig(basis.mean(axis=(0,1,2)))
        print(w)
        print(meanEnergy[-1])


# print("asdasdasdasdasdasd")

# np.random.seed(42)
# initializeRandomQuaternions(orientation)
# orientation = np.pad(orientation, 1, mode='wrap')[..., 1:-1]
# ind = ind.reshape(L,L,L,3)

# for _ in range(100):
#     for x,y,z in product(range(0,3), range(0,3), range(0,3)):
#         indexes = ind[x::3,y::3,z::3,:].reshape(-1,3)
#         doorientationSweep(orientation, indexes)
#     print(energy.mean(), basis.mean(axis=(0,1,2)))
