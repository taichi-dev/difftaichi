import taichi as ti
import sys
import math
import numpy as np
import os
import matplotlib.pyplot as plt

real = ti.f32
ti.init(default_fp=real, flatten_if=True)

max_steps = 2048
vis_interval = 64
output_vis_interval = 16
steps = 1024
assert steps * 2 <= max_steps

vis_resolution = 1024

scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(2, dtype=real)

loss = scalar()

init_x = vec()
init_v = vec()

x = vec()
x_inc = vec()  # for TOI
v = vec()
impulse = vec()

billiard_layers = 4
n_balls = 1 + (1 + billiard_layers) * billiard_layers // 2
target_ball = n_balls - 1
# target_ball = 0
goal = [0.9, 0.75]
radius = 0.03
elasticity = 0.8

ti.root.dense(ti.i, max_steps).dense(ti.j, n_balls).place(x, v, x_inc, impulse)
ti.root.place(init_x, init_v)
ti.root.place(loss)
ti.root.lazy_grad()

dt = 0.003
alpha = 0.00000
learning_rate = 0.01


@ti.func
def collide_pair(t, i, j):
    imp = ti.Vector([0.0, 0.0])
    x_inc_contrib = ti.Vector([0.0, 0.0])
    if i != j:
        dist = (x[t, i] + dt * v[t, i]) - (x[t, j] + dt * v[t, j])
        dist_norm = dist.norm()
        rela_v = v[t, i] - v[t, j]
        if dist_norm < 2 * radius:
            dir = ti.Vector.normalized(dist)
            projected_v = dir.dot(rela_v)

            if projected_v < 0:
                imp = -(1 + elasticity) * 0.5 * projected_v * dir
                toi = (dist_norm - 2 * radius) / min(
                    -1e-3, projected_v)  # Time of impact
                x_inc_contrib = min(toi - dt, 0) * imp
    x_inc[t + 1, i] += x_inc_contrib
    impulse[t + 1, i] += imp


@ti.kernel
def collide(t: ti.i32):
    for i in range(n_balls):
        for j in range(i):
            collide_pair(t, i, j)
    for i in range(n_balls):
        for j in range(i + 1, n_balls):
            collide_pair(t, i, j)


@ti.kernel
def advance(t: ti.i32):
    for i in range(n_balls):
        v[t, i] = v[t - 1, i] + impulse[t, i]
        x[t, i] = x[t - 1, i] + dt * v[t, i] + x_inc[t, i]


@ti.kernel
def compute_loss(t: ti.i32):
    loss[None] = (x[t, target_ball][0] - goal[0])**2 + (x[t, target_ball][1] -
                                                        goal[1])**2


@ti.kernel
def initialize():
    x[0, 0] = init_x[None]
    v[0, 0] = init_v[None]


gui = ti.GUI("Billiards", (1024, 1024), background_color=0x3C733F)


def forward(visualize=False, output=None):
    initialize()

    interval = vis_interval
    if output:
        interval = output_vis_interval
        os.makedirs('billiards/{}/'.format(output), exist_ok=True)

    count = 0
    for i in range(billiard_layers):
        for j in range(i + 1):
            count += 1
            x[0, count] = [
                i * 2 * radius + 0.5, j * 2 * radius + 0.5 - i * radius * 0.7
            ]

    pixel_radius = int(radius * 1024) + 1

    for t in range(1, steps):
        collide(t - 1)
        advance(t)

        if (t + 1) % interval == 0 and visualize:
            gui.clear()
            gui.circle((goal[0], goal[1]), 0x00000, pixel_radius // 2)

            for i in range(n_balls):
                if i == 0:
                    color = 0xCCCCCC
                elif i == n_balls - 1:
                    color = 0x3344cc
                else:
                    color = 0xF20530

                gui.circle((x[t, i][0], x[t, i][1]), color, pixel_radius)

            if output:
                gui.show('billiards/{}/{:04d}.png'.format(output, t))
            else:
                gui.show()

    compute_loss(steps - 1)


@ti.kernel
def clear():
    for t, i in ti.ndrange(max_steps, n_balls):
        impulse[t, i] = ti.Vector([0.0, 0.0])
        x_inc[t, i] = ti.Vector([0.0, 0.0])


def optimize():
    init_x[None] = [0.1, 0.5]
    init_v[None] = [0.3, 0.0]

    clear()
    # forward(visualize=True, output='initial')

    for iter in range(200):
        clear()

        with ti.Tape(loss):
            if iter % 20 == 19:
                output = 'iter{:04d}'.format(iter)
            else:
                output = None
            forward(visualize=True, output=output)

        print('Iter=', iter, 'Loss=', loss[None])
        for d in range(2):
            init_x[None][d] -= learning_rate * init_x.grad[None][d]
            init_v[None][d] -= learning_rate * init_v.grad[None][d]

    clear()
    forward(visualize=True, output='final')


def scan(zoom):
    N = 1000
    angles = []
    losses = []
    forward(visualize=True, output='initial')
    for i in range(N):
        alpha = ((i + 0.5) / N - 0.5) * math.pi * zoom
        init_x[None] = [0.1, 0.5]
        init_v[None] = [0.3 * math.cos(alpha), 0.3 * math.sin(alpha)]

        loss[None] = 0
        clear()
        forward(visualize=False)
        print(loss[None])

        losses.append(loss[None])
        angles.append(math.degrees(alpha))

    plt.plot(angles, losses)
    fig = plt.gcf()
    fig.set_size_inches(5, 3)
    plt.title('Billiard Scene Objective')
    plt.ylabel('Objective')
    plt.xlabel('Angle of velocity')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        scan(float(sys.argv[1]))
    else:
        optimize()
