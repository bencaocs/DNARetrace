import torch
import torch.nn as nn
import numpy as np

class NaiveFourierKANLayer(nn.Module):
    def __init__(self, inputdim, outdim, gridsize=2):
        super(NaiveFourierKANLayer, self).__init__()
        self.gridsize = gridsize
        self.inputdim = inputdim
        self.outdim = outdim

        self.fouriercoeffs = nn.Parameter(torch.randn(2, outdim, inputdim, gridsize) /
                                          (np.sqrt(inputdim) * np.sqrt(self.gridsize)))
    def forward(self, x):
        xshp = x.shape
        outshape = xshp[0:-1] + (self.outdim,)
        x = x.view(-1, self.inputdim)
        # Starting at 1 because constant terms are in the bias
        k = torch.reshape(torch.arange(1, self.gridsize + 1, device=x.device), (1, 1, 1, self.gridsize))
        xrshp = x.view(x.shape[0], 1, x.shape[1], 1)
        # This should be fused to avoid materializing memory
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)
        c = torch.reshape(c, (1, x.shape[0], x.shape[1], self.gridsize))
        s = torch.reshape(s, (1, x.shape[0], x.shape[1], self.gridsize))
        y = torch.einsum("dbik,djik->bj", torch.concat([c, s], axis=0), self.fouriercoeffs)

        y = y.view(outshape)
        return y

# import torch as th
# import numpy as np

# class NaiveFourierKANLayer(th.nn.Module):
#     def __init__( self, inputdim, outdim, gridsize=1, addbias=True, smooth_initialization=False):
#         super(NaiveFourierKANLayer,self).__init__()
#         self.gridsize= gridsize
#         self.addbias = addbias
#         self.inputdim = inputdim
#         self.outdim = outdim
        
#         grid_norm_factor = (th.arange(gridsize) + 1)**2 if smooth_initialization else np.sqrt(gridsize)
        
#         self.fouriercoeffs = th.nn.Parameter( th.randn(2,outdim,inputdim,gridsize) / 
#                                                 (np.sqrt(inputdim) * grid_norm_factor ) )
#         if( self.addbias ):
#             self.bias  = th.nn.Parameter( th.zeros(1,outdim))

#     def forward(self,x):
#         xshp = x.shape
#         outshape = xshp[0:-1]+(self.outdim,)
#         x = th.reshape(x,(-1,self.inputdim))
#         k = th.reshape( th.arange(1,self.gridsize+1,device=x.device),(1,1,1,self.gridsize))
#         xrshp = th.reshape(x,(x.shape[0],1,x.shape[1],1) ) 
#         c = th.cos( k*xrshp )
#         s = th.sin( k*xrshp )
#         y =  th.sum( c*self.fouriercoeffs[0:1],(-2,-1)) 
#         y += th.sum( s*self.fouriercoeffs[1:2],(-2,-1))
#         if( self.addbias):
#             y += self.bias

#         y = th.reshape( y, outshape)
#         return y