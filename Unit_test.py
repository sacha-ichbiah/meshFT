import trimesh, torch
import numpy as np
from meshft import compute_box_size, Fourier3dMesh
from time import time

if torch.cuda.is_available():
    print("Computations done in CUDA!")
    device = 'cuda:0'
    n_subdivisions = 3
    box_shape = np.array([100]*3)
else :
    print("No CUDA devices available, Computations done in CPU")
    device = 'cpu'
    n_subdivisions = 1
    box_shape = np.array([40]*3)

Mesh = trimesh.primitives.Sphere(subdivisions = n_subdivisions)
faces = np.array(Mesh.faces)
verts = np.array(Mesh.vertices)*100
Verts = torch.tensor(verts,device = device, dtype = torch.float,requires_grad=True)
Faces = torch.tensor(faces,device = device, dtype = torch.long)

#Give the dimensions of the box
#box_size = np.array([[-1.2,  1.2],
#                     [-1.2,  1.2],
#                     [-1.2,  1.2]])
#Or compute it automatically with a given offset
box_size = compute_box_size(verts,offset=0.2)


meshFT = Fourier3dMesh(box_size,box_shape,device=device, dtype = torch.float32,sigma_base = 100,gaussian_filter=True,narrowband_thresh=0.01)
t1 = time()
ftmesh = meshFT(Verts,Faces)
t2 = time()
print("time forward pass", t2 - t1)
#Compute the backward pass
t1 = time()
loss = torch.sum(torch.abs(ftmesh))
loss.backward()
t2 = time()
print("time backward pass", t2 - t1)
print(Verts.grad)