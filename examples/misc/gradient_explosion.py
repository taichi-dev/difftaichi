import taichi as ti
import matplotlib.pyplot as plt
import math
import sys

real = ti.f32
ti.init(default_fp=real)

x = ti.field(dtype=ti.f32)
v = ti.field(dtype=ti.f32)
a = ti.field(dtype=ti.f32)
loss = ti.field(dtype=ti.f32)
damping = ti.field(dtype=ti.f32)

max_timesteps = 1024 * 1024

dt = 0.001


def allocate_fields():
    ti.root.dense(ti.i, max_timesteps).place(x, v)
    ti.root.place(a, damping, loss)
    ti.root.lazy_grad()


@ti.kernel
def advance(t: ti.i32):
    v[t] = damping[None] * v[t - 1] + a[None]
    x[t] = x[t - 1] + dt * v[t]


@ti.kernel
def compute_loss(t: ti.i32):
    loss[None] = x[t]


def gradient(alpha, num_steps):
    damping[None] = math.exp(-dt * alpha)
    a[None] = 1
    with ti.Tape(loss):
        for i in range(1, num_steps):
            advance(i)
        compute_loss(num_steps - 1)
    return loss[None]


large = False
if len(sys.argv) > 1:
    large = True

allocate_fields()

# c = ['r', 'g', 'b', 'y', 'k']
for i, alpha in enumerate([0, 1, 3, 10]):
    xs, ys = [], []
    grads = []
    for num_steps in range(0, 10000 if large else 1000, 50):
        g = gradient(alpha, num_steps)
        xs.append(num_steps)
        ys.append(g)
    plt.plot(xs, ys, label="damping={}".format(alpha))

# plt.loglog()
fig = plt.gcf()
fig.set_size_inches(5, 3)
plt.title("Gradient Explosion without Damping")
plt.ylabel("Gradient")
plt.xlabel("Time steps")
plt.legend()
if large:
    plt.ylim(0, 3000)
plt.tight_layout()

plt.show()
