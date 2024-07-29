'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-07-04 20:31:59
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-07-04 20:32:01
 # @ Description: This file is distributed under the MIT license.
 '''

import taichi as ti
import taichi.math as tm
ti.init()

width = 500
height = 500

"""typing system of the complex matrices and variables"""
init_center = (250,232)
init_variance = (200.0, 200.0)

weyl_field = ti.field(ti.f32, shape = (width, height, 2) )

@ti.func
def calculate_density():
    for x, y, c in weyl_field:
        weyl_field[x,y] = weyl_field[x, y]


@ti.kernel
def init_weyl_field():
    volume = 1000/tm.sqrt(2 * tm.pi * init_variance[0] * init_variance[1])
    var_x, var_y = init_variance
    cx, cy = init_center
    for x,y,c in weyl_field:
        weyl_field[x,y,c] = volume * tm.exp( -tm.pow(x - cx,2)/var_x - tm.pow(y - cy,2)/var_y)
    #calculate_density()

if __name__ == "__main__":
    gui = ti.GUI("Weyl Spinor", res = (width, height))
    init_weyl_field()
    while gui.running:
        gui.set_image(weyl_field[:,:,0])
        gui.show()