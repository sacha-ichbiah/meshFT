import torch
import numpy as np
from torch import nn

def compute_box_size(verts,offset=0.1):
    #Compute automatically the box according to the position of the vertices. 
    #An offset of 0.1 means 10% of excess size. 
    
    box_size = np.zeros((3,2))
    extent = verts.max(axis=0)-verts.min(axis=0)
    box_size[:,1]=verts.max(axis=0)+extent*offset/2
    box_size[:,0]=verts.min(axis=0)-extent*offset/2
    box_size[:,0]=box_size[:,0].min(axis=0)
    box_size[:,1]=box_size[:,1].max(axis=0)
    return(box_size)

def get_internal_triangle_and_mesh_areas(Verts,Faces):
    Coords = Verts[Faces]
    cross_prods = torch.cross(Coords[:,1]-Coords[:,0],Coords[:,2]-Coords[:,0],dim=1)
    trisar =  0.5*torch.norm(cross_prods,dim=1)
    meshar = torch.sum(trisar)
    return(trisar,meshar)

def fourier3dfunctionPy(Verts,Faces,xi0, xi1, xi2):
    #This is the real surrogate of our torch function in python.
    trisar,meshar = get_internal_triangle_and_mesh_areas(Verts,Faces)

    tom = 2/meshar
    trisar*=tom

    v0 = Faces[:,0]
    v1 = Faces[:,1]
    v2 = Faces[:,2]

    Gridxi0, Gridxi1, Gridxi2 = torch.meshgrid(xi0,xi1,xi2,indexing = 'ij')
    Gridxi0 = Gridxi0.to(xi0.device)
    Gridxi1.to(xi0.device)
    Gridxi2.to(xi1.device)
    
    xiv =  (Gridxi0.reshape(-1,1)@(Verts[:,0].reshape(1,-1)) +
              Gridxi1.reshape(-1,1)@(Verts[:,1].reshape(1,-1)) + 
              Gridxi2.reshape(-1,1)@(Verts[:,2].reshape(1,-1)))

    emxiv = torch.exp(-1j* xiv)

    eps = 1.19209e-07 #For double:2.22045e-16     For float: 1.19209e-07
    Seps = np.sqrt(eps)

    xia = xiv[:,v0]
    xib = xiv[:,v1]
    xic = xiv[:,v2]
    emixia = emxiv[:,v0]
    emixib = emxiv[:,v1]
    emixic = emxiv[:,v2]
    xiab = xib-xia
    xibc = xic-xib
    xica = xia-xic

    C1 = ((torch.abs(xiab)<Seps).type(torch.long) * (torch.abs(xibc)<Seps).type(torch.long) + 
    (torch.abs(xibc)<Seps).type(torch.long) * (torch.abs(xica)<Seps).type(torch.long) +
    (torch.abs(xica)<Seps).type(torch.long) * (torch.abs(xiab)<Seps).type(torch.long)).clip(0,1)
    C2 = (torch.abs(xiab)<=Seps).type(torch.long)
    C3 = (torch.abs(xibc)<=Seps).type(torch.long)
    C4 = (torch.abs(xica)<=Seps).type(torch.long)

    R1 = C1
    R2 = C2 * (1-R1)
    R3 = C3 * (1-R2)*(1-R1)
    R4 = C4 * (1-R3)*(1-R2)*(1-R1)
    R5 = torch.ones_like(xiab).type(torch.long) * (1-R4)*(1-R3)*(1-R2)*(1-R1)

    R1=R1.type(torch.bool)
    R2=R2.type(torch.bool)
    R3=R3.type(torch.bool)
    R4=R4.type(torch.bool)
    R5=R5.type(torch.bool)


    Grid_results = torch.zeros_like(emixia)

    Grid_results[R1] = torch.exp(-1j*((xia[R1]+xib[R1]+xic[R1])/3))/2
    Grid_results[R2] = (1j*torch.exp(-1j*((xia[R2]+xib[R2])/2))+(torch.exp(-1j*((xia[R2]+xib[R2])/2))-emixic[R2])/((xia[R2]+xib[R2])/2-xic[R2]))/((xia[R2]+xib[R2])/2-xic[R2])
    Grid_results[R3] = (1j*torch.exp(-1j*((xib[R3]+xic[R3])/2))+(torch.exp(-1j*((xib[R3]+xic[R3])/2))-emixia[R3])/((xib[R3]+xic[R3])/2-xia[R3]))/((xib[R3]+xic[R3])/2-xia[R3])
    Grid_results[R4] = (1j*torch.exp(-1j*((xic[R4]+xia[R4])/2))+(torch.exp(-1j*((xic[R4]+xia[R4])/2))-emixib[R4])/((xic[R4]+xia[R4])/2-xib[R4]))/((xic[R4]+xia[R4])/2-xib[R4])
    Grid_results[R5] = emixia[R5]/(xiab[R5]*xica[R5])+emixib[R5]/(xibc[R5]*xiab[R5])+emixic[R5]/(xica[R5]*xibc[R5])


    #ftmesh = torch.complex(torch.zeros_like(Gridxi0),torch.zeros_like(Gridxi0))
    trisar_complex = torch.complex(trisar,torch.zeros_like(trisar))
    ftmesh= (Grid_results@(trisar_complex)).reshape(Gridxi0.shape)
    return(ftmesh)

class Fourier3dFunctionPy(nn.Module):
    
    """
    Module for the meshFT layer. Takes in a triangle mesh and returns a fourier transform.
    """
    def __init__(self, box_size,box_shape,device = 'cpu', dtype = torch.float):
        """
        box_shape: [x_res,y_res,z_res] Size of the fourier box (in voxels)
        box_size: [[x_min,xmax],[y_min,y_max],[z_min,z_max]] Size of the box (in the spatial dimensions of the mesh)"""

        super().__init__()
        self.box_size = torch.tensor(box_size,device = device, dtype = dtype)
        self.box_shape = box_shape
        self.dtype = dtype
        self.device = device
        self.xi0,self.xi1,self.xi2 = self._compute_spatial_frequency_grid()
        

    def forward(self, Verts,Faces):
        """
        Verts: vertex tensor. float tensor of shape (n_vertex, 3)
        Faces: faces tensor. int tensor of shape (n_faces, 3)
                  if j cols, triangulate/tetrahedronize interior first.
        return meshFT: complex fourier transform of the mesh of shape self.box_shape
        """
        ftmesh = fourier3dfunctionPy(Verts-self.box_size[:,0],Faces,self.xi0, self.xi1, self.xi2)
                        
        return ftmesh
    
    def _compute_spatial_frequency_grid(self): 
        n0,n1,n2 = self.box_shape
        nn0,nn1,nn2 = n0-1,n1-1,n2-1

        xi0 = torch.zeros(n0,dtype = self.dtype,device = self.device)
        xi1 = torch.zeros(n1,dtype = self.dtype,device = self.device)
        xi2 = torch.zeros(n2,dtype = self.dtype,device = self.device)

        s0,s1,s2 = np.pi/(self.box_size[0,1]-self.box_size[0,0]),np.pi/(self.box_size[1,1]-self.box_size[1,0]),np.pi/(self.box_size[2,1]-self.box_size[2,0])

        K0 = torch.arange(n0,dtype = self.dtype,device = self.device)
        Kn0 = 2*K0-nn0
        xi0 = Kn0*s0
        K1 = torch.arange(n1,dtype = self.dtype,device = self.device)
        Kn1 = 2*K1-nn1
        xi1 = Kn1*s1
        K2 = torch.arange(n2,dtype = self.dtype,device = self.device)
        Kn2 = 2*K2-nn2
        xi2 = Kn2*s2
        return(xi0,xi1,xi2)
    
if not torch.cuda.is_available():
    Fourier3dMesh = Fourier3dFunctionPy#()
else : 
    from .cuda_class import Fourier3dFunctionCUDA
    Fourier3dMesh = Fourier3dFunctionCUDA#()
