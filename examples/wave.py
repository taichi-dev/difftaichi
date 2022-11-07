import taichi as ti
import argparse
import math
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

real = ti.f32
ti.init(default_fp=real, arch=ti.cuda)

n_grid = 256
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 3e-4
max_steps = 512
vis_interval = 32
output_vis_interval = 2
steps = 256
assert steps * 2 <= max_steps
amplify = 1

scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector(2, dtype=real)

p = scalar()
target = scalar()
initial = scalar()
loss = scalar()


def allocate_fields():
    ti.root.dense(ti.i, max_steps).dense(ti.jk, n_grid).place(p)
    ti.root.dense(ti.i, max_steps).dense(ti.jk, n_grid).place(p.grad)
    ti.root.dense(ti.ij, n_grid).place(target)
    ti.root.dense(ti.ij, n_grid).place(target.grad)
    ti.root.dense(ti.ij, n_grid).place(initial)
    ti.root.dense(ti.ij, n_grid).place(initial.grad)
    ti.root.place(loss)
    ti.root.place(loss.grad)


c = 340
# damping
alpha = 0.00000
inv_dx2 = inv_dx * inv_dx
dt = (math.sqrt(alpha * alpha + dx * dx / 3) - alpha) / c
learning_rate = 1


@ti.func
def get_p(t, i, j):
    return p[t, i, j] if 0 <= i < n_grid and 0 <= j < n_grid else 0


@ti.func
def laplacian(t, i, j):
    return inv_dx2 * (-4 * get_p(t, i, j) + get_p(t, i, j - 1) + get_p(t, i, j + 1) +
                      get_p(t, i + 1, j) + get_p(t, i - 1, j))


@ti.kernel
def initialize():
    for i in range(n_grid):
        for j in range(n_grid):
            p[0, i, j] = initial[i, j]


@ti.kernel
def fdtd(t: ti.i32):
    for i in range(n_grid):  # Parallelized over GPU threads
        for j in range(n_grid):
            laplacian_p = laplacian(t - 2, i, j)
            laplacian_q = laplacian(t - 1, i, j)
            p[t, i, j] = 2 * p[t - 1, i, j] + (
                c * c * dt * dt + c * alpha * dt) * laplacian_q - p[
                    t - 2, i, j] - c * alpha * dt * laplacian_p


@ti.kernel
def compute_loss(t: ti.i32):
    for i in range(n_grid):
        for j in range(n_grid):
            loss[None] += dx * dx * (target[i, j] - p[t, i, j])**2


@ti.kernel
def apply_grad():
    # gradient descent
    for i, j in initial.grad:
        initial[i, j] -= learning_rate * initial.grad[i, j]

@ti.ad.no_grad
@ti.kernel
def get_image(img: ti.types.ndarray(), t: ti.i32):
    for i, j in ti.ndrange(n_grid, n_grid):
        img[i, j] = p[t, i, j] * amplify + 0.5


def forward(output=None):
    steps_mul = 1
    interval = vis_interval
    if output:
        os.makedirs(output, exist_ok=True)
        steps_mul = 2
        interval = output_vis_interval
    initialize()
    for t in range(2, steps * steps_mul):
        fdtd(t)
        if (t + 1) % interval == 0:
            img = np.zeros(shape=(n_grid, n_grid), dtype=np.float32)
            get_image(img, t)
            img = cv2.resize(img, fx=4, fy=4, dsize=None)
            cv2.imshow('img', img)
            cv2.waitKey(1)
            if output:
                img = np.clip(img, 0, 255)
                cv2.imwrite(output + "/{:04d}.png".format(t), img * 255)
    compute_loss(steps - 1)


@ti.kernel
def fill_target(img: ti.types.ndarray()):
    for i, j in ti.ndrange(n_grid, n_grid):
        target[i, j] = float(img[i, j])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=200)
    options = parser.parse_args()

    # initialization
    target_img = cv2.imread('taichi.png')[:, :, 0] / 255.0
    target_img -= target_img.mean()
    target_img = cv2.resize(target_img, (n_grid, n_grid))
    cv2.imshow('target', target_img * amplify + 0.5)
    allocate_fields()
    # print(target_img.min(), target_img.max())
    fill_target(target_img)

    if False:
        # this is not too exciting...
        initial[n_grid // 2, n_grid // 2] = -2
        forward('center')
        initial[n_grid // 2, n_grid // 2] = 0

    for opt in range(options.iters):
        with ti.ad.Tape(loss):
            output = None
            if opt % 20 == 19:
                output = 'wave/iter{:03d}/'.format(opt)
            forward(output)

        print('Iter', opt, ' Loss =', loss[None])

        apply_grad()

    forward('optimized')


if __name__ == '__main__':
    main()
