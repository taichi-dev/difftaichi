import sys
import taichi as ti
import math
import numpy as np
import os
import matplotlib.pyplot as plt
import time
from matplotlib.pyplot import cm
import sys

real = ti.f32
ti.init(default_fp=real)

max_steps = 4096
vis_interval = 1
output_vis_interval = 1
steps = 2000
assert steps * 2 <= max_steps

vis_resolution = 1024

scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(2, dtype=real)

loss = scalar()

x = vec()
v = vec()

n_objects = 1
ground_height = 0.1


def allocate_fields():
    ti.root.dense(ti.l, max_steps).dense(ti.i, n_objects).place(x, v)
    ti.root.place(loss)
    ti.root.lazy_grad()


total_t = 0.35
dt = total_t / steps
learning_rate = 1.0

use_toi = False


@ti.kernel
def advance_toi(t: ti.i32):
    for i in range(n_objects):
        old_v = v[t - 1, i]
        new_v = old_v
        old_x = x[t - 1, i]
        new_x = old_x + dt * new_v
        toi = 0.0
        if new_x[1] < ground_height and new_v[1] < 0:
            new_v[1] = -new_v[1]
            toi = -(old_x[1] - ground_height) / old_v[1]
        v[t, i] = new_v
        x[t, i] = x[t - 1, i] + toi * old_v + (dt - toi) * new_v


@ti.kernel
def advance_no_toi(t: ti.i32):
    for i in range(n_objects):
        new_v = v[t - 1, i]
        if x[t - 1, i][1] < ground_height and new_v[1] < 0:
            new_v[1] = -new_v[1]
        v[t, i] = new_v
        x[t, i] = x[t - 1, i] + dt * new_v


gui = ti.GUI("Rigid Body", (1024, 1024), background_color=0xFFFFFF)


def forward(output=None, visualize=True, dy=0, i=0):
    x[0, 0] = [0.8, 0.4 + dy]
    v[0, 0] = [-2, -2]

    interval = vis_interval
    total_steps = steps
    if output:
        os.makedirs('rigid_body_toi/{}/'.format(output), exist_ok=True)

    gui.clear()
    for t in range(1, total_steps):
        if use_toi:
            advance_toi(t)
        else:
            advance_no_toi(t)

        if (t + 1) % interval == 0 and visualize:
            color = 0x010101 * min(
                255, max(0, int((1 - t * dt / total_t) * 0.7 * 255)))
            gui.circle((x[t, 0][0], x[t, 0][1]), color, 80)
            offset = 0.077
            gui.line((0.05, ground_height - offset),
                     (0.95, ground_height - offset), color=0x000000, radius=2)

    if output:
        gui.show('rigid_body_toi/{}/{:04d}.png'.format(output, i))
    else:
        gui.show()


def main():
    allocate_fields()
    if len(sys.argv) != 2:
        print('Usage: python3 script.py [use_toi=0/1] [steps=100]')
    global steps
    global use_toi
    use_toi = bool(int(sys.argv[1]))
    print(use_toi)
    steps = int(sys.argv[2])
    global dt
    dt = total_t / steps
    for i, dy in enumerate(np.arange(0, 0.2, 0.001)):
        forward(visualize=True, dy=dy, output='animation', i=i)


if __name__ == '__main__':
    main()
