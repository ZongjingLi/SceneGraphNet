'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-07-04 21:20:25
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-07-04 21:20:26
 # @ Description: This file is distributed under the MIT license.
 '''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

dt = .005
rx, ry = 5., 5.
div = 250
div_x, div_y = (div,div)

cx, cy = 1.0, 0.5
cx2, cy2 = -1.0, -2.5
vx, vy = .01, .01

x = torch.linspace(-rx, rx, div_x)
y = torch.linspace(-ry, ry, div_y)

grid_x, grid_y = torch.meshgrid([x, y])
coords = torch.cat([grid_x[...,None], grid_y[...,None]], dim = -1)

N = div_x
dx = rx * 2 / div_x
diff = (np.zeros((N), np.float32)
            + np.diag(np.ones((N-1), np.float32), 1)
            -np.diag(np.ones((N-1), np.float32), -1))/(dx * 2)
diff = torch.tensor(diff)



def init_weyl_field_gaussian(coords, center, variance):
    cx, cy = center
    var_x, var_y = variance
    volume = 1/math.sqrt(2 * torch.pi * var_x * var_y)
    coords_x = coords[:, :, 0]
    coords_y = coords[:, :, 1]
    field = volume * torch.exp( 
        -torch.pow(coords_x - cx,2)/var_x
        -torch.pow(coords_y - cy,2)/var_y).unsqueeze(-1)
    return torch.cat([field, torch.zeros_like(field)], dim = -1)

def init_weyl_field_gaussian2(coords, center, variance):
    cx, cy = center
    var_x, var_y = variance
    volume = 1/math.sqrt(2 * torch.pi * var_x * var_y)
    coords_x = coords[:, :, 0]
    coords_y = coords[:, :, 1]
    field = volume * torch.exp( 
        -torch.pow(coords_x - cx,2)/var_x
        -torch.pow(coords_y - cy,2)/var_y).unsqueeze(-1)
    return torch.cat([torch.zeros_like(field),field], dim = -1)

def density(field):
    return torch.sqrt(field[:,:,0] ** 2 + field[:,:,1] ** 2)

sigma_x = torch.tensor(
    [[0.0, 1.0],
     [1.0, 0.0]])
sigma_z = torch.tensor(
    [[1.0, 0.0],
     [0.0, -1.0]])

def simulate(field, dt = 0.01):

    gradient_field_x = torch.matmul(diff,field[:,:,0])
    gradient_field_y = torch.matmul(diff,field[:,:,1])
    gradient = torch.cat([gradient_field_x[...,None], gradient_field_y[...,None]], dim = -1)
    return field \
        + dt *  torch.einsum("dc,whd->whd",sigma_x,gradient)\
        + dt * torch.einsum("dc,whd->whd",sigma_z,gradient)

if __name__ == "__main__":
    plt.figure("diff", figsize = (4,4))
    x = torch.linspace(-3,3, div_x)
    f = torch.sin(x)
    df = torch.einsum("hh,h->h",diff, f)
    df = torch.matmul(diff,f)
    plt.plot(f)
    plt.plot(df)    

    plt.figure("diff2", figsize = (4,4))
    field = torch.sin(coords)
    field = init_weyl_field_gaussian(coords, [cx, cy], [vx, vy])
    plt.subplot(121)
    gradient_field_x = torch.matmul(diff,field[:,:,0])
    plt.imshow(gradient_field_x)
    plt.subplot(122)
    gradient_field_y = torch.matmul(diff,field[:,:,1])
    plt.imshow(gradient_field_y)
    

    plt.figure("weyl", figsize = (8,8))
    field = init_weyl_field_gaussian(coords, [cx, cy], [vx, vy])\
          + init_weyl_field_gaussian2(coords, [cx2, cy2], [vx, vy])
    for t in range(1000):
        field = simulate(field, dt = dt)
        plt.imshow(density(field))
        plt.pause(0.01)
        plt.cla()
    plt.show()