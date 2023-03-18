import math
from torch import nn
from torch.autograd import Function
import torch

from .fourier3d_cpp import forward_fourier3d_cpp,backward_fourier3d_cpp

torch.manual_seed(42)


class MeshFTCUDA(Function):
    
    @staticmethod
    def forward(ctx, Verts, Faces, xi0,xi1,xi2):

        assert Verts.dtype in (torch.float, torch.double) , "Mesh FT not implemented for " + str(Verts.dtype)
        Gridxi0, Gridxi1, Gridxi2 = torch.meshgrid(xi0,xi1,xi2,indexing = 'ij')
        xi_list = torch.vstack((Gridxi0.reshape(-1),Gridxi1.reshape(-1),Gridxi2.reshape(-1))).transpose(0,1).contiguous()
        Faces_coeff = torch.ones(len(Faces),device = Faces.device)
        ftmesh_list = forward_fourier3d_cpp(Verts, Faces, Faces_coeff, xi_list).contiguous()
        ctx.save_for_backward(Verts, Faces,Faces_coeff, xi_list,ftmesh_list)
        ftmesh = torch.complex(torch.zeros_like(Gridxi0),torch.zeros_like(Gridxi0))
        return ftmesh

    @staticmethod
    def backward(ctx, grad_output):
        Verts, Faces, Faces_coeff,xi_list,ftmesh_list = ctx.saved_tensors
        grad_output_list = (grad_output.reshape(-1)).contiguous()
        gradVerts = backward_fourier3d_cpp(Verts, Faces, Faces_coeff, grad_output_list, ftmesh_list, xi_list)
        return gradVerts, None, None, None, None



class Fourier3dFunctionCUDA(nn.Module):
    
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
        ftmesh = MeshFTCUDA(Verts-self.box_size[:,0],Faces,self.xi0, self.xi1, self.xi2)
                        
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