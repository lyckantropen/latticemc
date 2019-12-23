import numpy as np
from itertools import product
from numba import jit,njit,int32,float32

@njit
def randomQuaternion(r):
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

    result[0]=r*y1;
    result[1]=r*y2;
    result[2]=r*y3*sr;
    result[3]=r*y4*sr;
    return result

@njit
def randomUniformQuaternion():
    return randomQuaternion(1)

@njit
def initializeRandomQuaternions(lattice):
    for i in np.ndindex(lattice.shape[:-1]):
        lattice[i] = randomUniformQuaternion()

@njit
def getOrientation(x):
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
def getNeighbors(center, lattice):
    ind = np.mod(np.array([
        [ 1,0,0],
        [-1,0,0],
        [0, 1,0],
        [0,-1,0],
        [0,0, 1],
        [0,0,-1]
    ]) + center, lattice.shape[0])
    return np.array([
        list(lattice[ind[0,0], ind[0,1], ind[0,2]]),
        list(lattice[ind[1,0], ind[1,1], ind[1,2]]),
        list(lattice[ind[2,0], ind[2,1], ind[2,2]]),
        list(lattice[ind[3,0], ind[3,1], ind[3,2]]),
        list(lattice[ind[4,0], ind[4,1], ind[4,2]]),
        list(lattice[ind[5,0], ind[5,1], ind[5,2]])
    ], np.float32)


@njit([float32(float32[:],float32[:,:],float32)])
def getEnergy(x, neighbors, t):
    e = getOrientation(x)
    energy = 0
    for i in range(neighbors.shape[0]):
        en = getOrientation(neighbors[i])
        energy += np.sum(e*en)
    energy /= -t
    return energy

# particle = np.dtype({
#     'names': ['index','x'],
#     'formats': [ (np.int, (3,)), (np.float32, (4,))]
# })

np.random.seed(42)
L = 9
#lattice = np.zeros((L,L,L), dtype=particle)
lattice = np.zeros((L,L,L,4), dtype=np.float32)
lat_energy = np.zeros((L,L,L), dtype=np.float32)
lat_e = np.zeros((L,L,L,3,3), dtype=np.float32)
initializeRandomQuaternions(lattice)
T = 2
r = 0.1
mean_energy = np.array([0], np.float32)
r_values = np.array([r], np.float32)

@jit(forceobj=True,nopython=False,parallel=True)
def process(lattice, indexes):
    global lat_energy, lat_e, r
    for _i in indexes:
        i=tuple(_i)
        x = lattice[i]
        #nind = getNeighborsIndices(_i, L)
        neighbors = getNeighbors(_i, lattice)
        energy = getEnergy(x, neighbors, T)
        # adjust x
        nx = x.copy()
        dx = randomQuaternion(r)
        nx += dx
        nx /= np.linalg.norm(nx)
        # evaluate change
        energy2 = getEnergy(nx, neighbors, T)
        if energy2<energy:
            lattice[i] = nx
            lat_energy[i] = energy2
        else:
            if np.random.random() < np.exp(-2*(energy2-energy)):
                lattice[i] = nx
                lat_energy[i] = energy2
        lat_e[i] = getOrientation(lattice[i])
    

ind = np.array(list(np.ndindex(lattice.shape[:-1])))
# lattice = np.pad(lattice, 1, mode='wrap')[..., 1:-1]
# ind = ind.reshape(L,L,L,3)

# @jit(forceobj=True,nopython=False,parallel=True)
# def process_subcells(ind, lattice):
#     for x,y,z in product(range(0,3), range(0,3), range(0,3)):
#         indexes = ind[x::3,y::3,z::3,:].reshape(-1,3)
#         process(lattice, indexes)

#from joblib import Parallel, delayed
#def process_subcells_joblib(ind, lattice, parallel):
#    parallel(delayed(process)(lattice, ind[x::3,y::3,z::3,:].reshape(-1,3)) for x,y,z in product(range(0,3), range(0,3), range(0,3)))

#with Parallel(n_jobs=27, prefer='threads') as parallel:

for it in range(10000):
    indexes = ind[np.random.randint(0, ind.shape[0], ind.shape[0])]
    process(lattice, indexes)
    #process_subcells(ind, lattice)
    #process_subcells_joblib(ind, lattice)
    mean_energy = np.append(mean_energy,lat_energy.mean())
    r_values = np.append(r_values, r)
    if mean_energy.size > 11:
        de = np.diff(mean_energy, 1)
        dr = np.diff(r_values, 1)
        efirst = de/dr
        rfirst = 0.5*(r_values[:-1]+ r_values[1:])
        d2e = np.diff(efirst, 1)
        dr2 = np.diff(rfirst, 1)
        rsecond = d2e/dr2
        if rsecond[-10].mean() < 0:
            r *= 1.01
        else:
            r *= 0.99
    r *= 1 + np.random.normal(scale=0.01)

    if it % 20 == 0 and it > 0:
        r = np.abs(np.random.normal(scale=0.5))
    if it % 50 == 0:
        if it>200:
            de = mean_energy[-10].mean() - mean_energy[-200:-180].mean()
            print(de)
        print(r)
        w,v = np.linalg.eig(lat_e.mean(axis=(0,1,2)))
        print(w)
        print(mean_energy[-1])


# print("asdasdasdasdasdasd")

# np.random.seed(42)
# initializeRandomQuaternions(lattice)
# lattice = np.pad(lattice, 1, mode='wrap')[..., 1:-1]
# ind = ind.reshape(L,L,L,3)

# for _ in range(100):
#     for x,y,z in product(range(0,3), range(0,3), range(0,3)):
#         indexes = ind[x::3,y::3,z::3,:].reshape(-1,3)
#         process(lattice, indexes)
#     print(lat_energy.mean(), lat_e.mean(axis=(0,1,2)))
