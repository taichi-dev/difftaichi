import taichi as ti
import os
import numpy as np
import cv2

real = ti.f32
ti.init(default_fp=real, arch=ti.cuda, flatten_if=True)

num_iterations = 240
n_grid = 128
dx = 1.0 / n_grid
num_iterations_gauss_seidel = 10
p_dims = num_iterations_gauss_seidel + 1
steps = 100
learning_rate = 400

scalar = lambda: ti.field(dtype=real)
vector = lambda: ti.Vector.field(2, dtype=real)

v = vector()
div = scalar()
p = scalar()
v_updated = vector()
target = scalar()
smoke = scalar()
loss = scalar()

ti.root.dense(ti.i, steps * p_dims).dense(ti.jk, n_grid).place(p)
ti.root.dense(ti.i, steps).dense(ti.jk, n_grid).place(v, v_updated, smoke, div)
ti.root.dense(ti.ij, n_grid).place(target)
ti.root.place(loss)
ti.root.lazy_grad()


# Integer modulo operator for positive values of n
@ti.func
def imod(n, divisor):
    ret = 0
    if n > 0:
        ret = n - divisor * (n // divisor)
    else:
        ret = divisor + n - divisor * (-n // divisor)
    return ret


@ti.func
def dec_index(index):
    new_index = index - 1
    if new_index < 0:
        new_index = n_grid - 1
    return new_index


@ti.func
def inc_index(index):
    new_index = index + 1
    if new_index >= n_grid:
        new_index = 0
    return new_index


@ti.kernel
def compute_div(t: ti.i32):
    for y in range(n_grid):
        for x in range(n_grid):
            div[t, y, x] = -0.5 * dx * (v_updated[t, inc_index(y), x][0] -
                                        v_updated[t, dec_index(y), x][0] +
                                        v_updated[t, y, inc_index(x)][1] -
                                        v_updated[t, y, dec_index(x)][1])


@ti.kernel
def compute_p(t: ti.i32, k: ti.i32):
    for y in range(n_grid):
        for x in range(n_grid):
            a = k + t * num_iterations_gauss_seidel
            p[a + 1, y,
              x] = (div[t, y, x] + p[a, dec_index(y), x] +
                    p[a, inc_index(y), x] + p[a, y, dec_index(x)] +
                    p[a, y, inc_index(x)]) / 4.0


@ti.kernel
def update_v(t: ti.i32):
    for y in range(n_grid):
        for x in range(n_grid):
            a = num_iterations_gauss_seidel * t - 1
            v[t, y, x][0] = v_updated[t, y, x][0] - 0.5 * (
                p[a, inc_index(y), x] - p[a, dec_index(y), x]) / dx
            v[t, y, x][1] = v_updated[t, y, x][1] - 0.5 * (
                p[a, y, inc_index(x)] - p[a, y, dec_index(x)]) / dx


@ti.kernel
def advect(field: ti.template(), field_out: ti.template(),
           t_offset: ti.template(), t: ti.i32):
    """Move field smoke according to x and y velocities (vx and vy)
     using an implicit Euler integrator."""
    for y in range(n_grid):
        for x in range(n_grid):
            center_x = y - v[t + t_offset, y, x][0]
            center_y = x - v[t + t_offset, y, x][1]

            # Compute indices of source cell
            left_ix = ti.cast(ti.floor(center_x), ti.i32)
            top_ix = ti.cast(ti.floor(center_y), ti.i32)

            rw = center_x - left_ix  # Relative weight of right-hand cell
            bw = center_y - top_ix  # Relative weight of bottom cell

            # Wrap around edges
            # TODO: implement mod (%) operator
            left_ix = imod(left_ix, n_grid)
            right_ix = left_ix + 1
            right_ix = imod(right_ix, n_grid)
            top_ix = imod(top_ix, n_grid)
            bot_ix = top_ix + 1
            bot_ix = imod(bot_ix, n_grid)

            # Linearly-weighted sum of the 4 surrounding cells
            field_out[t, y, x] = (1 - rw) * (
                (1 - bw) * field[t - 1, left_ix, top_ix] +
                bw * field[t - 1, left_ix, bot_ix]) + rw * (
                    (1 - bw) * field[t - 1, right_ix, top_ix] +
                    bw * field[t - 1, right_ix, bot_ix])


@ti.kernel
def compute_loss():
    for i in range(n_grid):
        for j in range(n_grid):
            loss[None] += (target[i, j] - smoke[steps - 1, i, j])**2 * (1 / n_grid**2)


@ti.kernel
def apply_grad():
    # gradient descent
    for i in range(n_grid):
        for j in range(n_grid):
            v[0, i, j] -= learning_rate * v.grad[0, i, j]


def forward(output=None):
    for t in range(1, steps):
        advect(v, v_updated, -1, t)

        compute_div(t)
        for k in range(num_iterations_gauss_seidel):
            compute_p(t, k)

        update_v(t)
        advect(smoke, smoke, 0, t)

        if output:
            smoke_ = np.zeros(shape=(n_grid, n_grid), dtype=np.float32)
            for i in range(n_grid):
                for j in range(n_grid):
                    smoke_[i, j] = smoke[t, i, j]
            cv2.imshow('smoke', smoke_)
            cv2.waitKey(1)
            os.makedirs(output, exist_ok=True)
            cv2.imwrite("{}/{:04d}.png".format(output, t), 255 * smoke_)
    compute_loss()


def main():
    target_img = cv2.resize(cv2.imread('taichi.png'),
                            (n_grid, n_grid))[:, :, 0] / 255.0

    for i in range(n_grid):
        for j in range(n_grid):
            target[i, j] = target_img[i, j]
            smoke[0, i, j] = (i // 16 + j // 16) % 2

    for opt in range(num_iterations):
        with ti.Tape(loss):
            output = "outputs/opt{:03d}".format(opt) if opt % 10 == 0 else None
            forward(output)
            velocity_field = np.ones(shape=(n_grid, n_grid, 3),
                                     dtype=np.float32)
            for i in range(n_grid):
                for j in range(n_grid):
                    s = 0.2
                    b = 0.5
                    velocity_field[i, j, 0] = v[0, i, j][0] * s + b
                    velocity_field[i, j, 1] = v[0, i, j][1] * s + b
            cv2.imshow('velocity', velocity_field)
            cv2.waitKey(1)

        print('Iter', opt, ' Loss =', loss[None])
        apply_grad()

    forward("output")


if __name__ == '__main__':
    main()
