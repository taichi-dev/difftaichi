import taichi as ti
import numpy as np
import cv2
import matplotlib.pyplot as plt

real = ti.f32
ti.init(default_fp=real, arch=ti.cuda)

dim = 2
n_particles = 6400
N = 80
n_grid = 120
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 3e-4
p_mass = 1
p_vol = 1
E = 100
mu = E
la = E
max_steps = 1024
steps = 1024
gravity = 9.8
target = [0.3, 0.6]

scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(dim, dtype=real)
mat = lambda: ti.Matrix.field(dim, dim, dtype=real)

x = ti.Vector.field(dim,
                    dtype=real,
                    shape=(max_steps, n_particles),
                    needs_grad=True)
x_avg = ti.Vector.field(dim, dtype=real, shape=(), needs_grad=True)
v = ti.Vector.field(dim,
                    dtype=real,
                    shape=(max_steps, n_particles),
                    needs_grad=True)
grid_v_in = ti.Vector.field(dim,
                            dtype=real,
                            shape=(n_grid, n_grid),
                            needs_grad=True)
grid_v_out = ti.Vector.field(dim,
                             dtype=real,
                             shape=(n_grid, n_grid),
                             needs_grad=True)
grid_m_in = ti.field(dtype=real, shape=(n_grid, n_grid), needs_grad=True)
C = ti.Matrix.field(dim,
                    dim,
                    dtype=real,
                    shape=(max_steps, n_particles),
                    needs_grad=True)
F = ti.Matrix.field(dim,
                    dim,
                    dtype=real,
                    shape=(max_steps, n_particles),
                    needs_grad=True)
init_v = ti.Vector.field(dim, dtype=real, shape=(), needs_grad=True)
loss = ti.field(dtype=real, shape=(), needs_grad=True)


@ti.kernel
def set_v():
    for i in range(n_particles):
        v[0, i] = init_v[None]


@ti.kernel
def clear_grid():
    for i, j in grid_m_in:
        grid_v_in[i, j] = [0, 0]
        grid_m_in[i, j] = 0
        grid_v_in.grad[i, j] = [0, 0]
        grid_m_in.grad[i, j] = 0
        grid_v_out.grad[i, j] = [0, 0]


@ti.kernel
def p2g(f: ti.i32):
    for p in range(0, n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_F = (ti.Matrix.diag(dim=2, val=1) + dt * C[f, p]) @ F[f, p]
        F[f + 1, p] = new_F
        J = (new_F).determinant()
        r, s = ti.polar_decompose(new_F)
        cauchy = 2 * mu * (new_F - r) @ new_F.transpose() + \
                 ti.Matrix.diag(2, la * (J - 1) * J)
        stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
        affine = stress + p_mass * C[f, p]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos = (ti.cast(ti.Vector([i, j]), real) - fx) * dx
                weight = w[i](0) * w[j](1)
                grid_v_in[base + offset] += weight * (p_mass * v[f, p] +
                                                      affine @ dpos)
                grid_m_in[base + offset] += weight * p_mass


bound = 3


@ti.kernel
def grid_op():
    for p in range(n_grid * n_grid):
        i = p // n_grid
        j = p - n_grid * i
        inv_m = 1 / (grid_m_in[i, j] + 1e-10)
        v_out = inv_m * grid_v_in[i, j]
        v_out[1] -= dt * gravity
        if i < bound and v_out[0] < 0:
            v_out[0] = 0
        if i > n_grid - bound and v_out[0] > 0:
            v_out[0] = 0
        if j < bound and v_out[1] < 0:
            v_out[1] = 0
        if j > n_grid - bound and v_out[1] > 0:
            v_out[1] = 0
        grid_v_out[i, j] = v_out


@ti.kernel
def g2p(f: ti.i32):
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, real)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector([0.0, 0.0])
        new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                dpos = ti.cast(ti.Vector([i, j]), real) - fx
                g_v = grid_v_out[base(0) + i, base(1) + j]
                weight = w[i](0) * w[j](1)
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx

        v[f + 1, p] = new_v
        x[f + 1, p] = x[f, p] + dt * v[f + 1, p]
        C[f + 1, p] = new_C


@ti.kernel
def compute_x_avg():
    for i in range(n_particles):
        x_avg[None].atomic_add((1 / n_particles) * x[steps - 1, i])


@ti.kernel
def compute_loss():
    dist = (x_avg[None] - ti.Vector(target))**2
    loss[None] = 0.5 * (dist(0) + dist(1))


@ti.ad.grad_replaced
def substep(s):
    clear_grid()
    p2g(s)
    grid_op()
    g2p(s)


@ti.ad.grad_for(substep)
def substep_grad(s):
    clear_grid()
    p2g(s)
    grid_op()

    g2p.grad(s)
    grid_op.grad()
    p2g.grad(s)


# initialization
init_v[None] = [0, 0]

for i in range(n_particles):
    F[0, i] = [[1, 0], [0, 1]]

for i in range(N):
    for j in range(N):
        x[0, i * N + j] = [dx * (i * 0.5 + 10), dx * (j * 0.5 + 25)]

set_v()

losses = []
img_count = 0
for i in range(30):
    with ti.Tape(loss=loss):
        set_v()
        for s in range(steps - 1):
            substep(s)

        loss[None] = 0
        x_avg[None] = [0, 0]
        compute_x_avg()
        compute_loss()
    l = loss[None]
    losses.append(l)
    grad = init_v.grad[None]
    print('loss=', l, '   grad=', (grad[0], grad[1]))
    learning_rate = 10
    init_v[None][0] -= learning_rate * grad[0]
    init_v[None][1] -= learning_rate * grad[1]

    # visualize
    for s in range(63, steps, 64):
        scale = 4
        img = np.zeros(shape=(scale * n_grid, scale * n_grid)) + 0.3
        total = [0, 0]
        for i in range(n_particles):
            p_x = int(scale * x[s, i][0] / dx)
            p_y = int(scale * x[s, i][1] / dx)
            total[0] += p_x
            total[1] += p_y
            img[p_x, p_y] = 1
        cv2.circle(img, (total[1] // n_particles, total[0] // n_particles),
                   radius=5,
                   color=0,
                   thickness=5)
        cv2.circle(
            img,
            (int(target[1] * scale * n_grid), int(target[0] * scale * n_grid)),
            radius=5,
            color=1,
            thickness=5)
        img = img.swapaxes(0, 1)[::-1]
        cv2.imshow('MPM', img)
        img_count += 1
        cv2.waitKey(1)

plt.title("Optimization of Initial Velocity")
plt.ylabel("Loss")
plt.xlabel("Gradient Descent Iterations")
plt.plot(losses)
plt.show()
