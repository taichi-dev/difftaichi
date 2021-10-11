import taichi as ti
import math
import numpy as np
import os
import matplotlib.pyplot as plt

real = ti.f32
ti.init(default_fp=real)

max_steps = 4096
vis_interval = 16
output_vis_interval = 16
steps = 2048
assert steps * 2 <= max_steps

vis_resolution = 1024

scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(2, dtype=real)

loss = scalar()

x = vec()
v = vec()
rotation = scalar()
omega = scalar()
friction = scalar()

halfsize = vec()

inverse_mass = scalar()
inverse_inertia = scalar()

v_inc = vec()
omega_inc = scalar()

head_id = 3
goal = [0.9, 0.15]

n_objects = 1
elasticity = 0.3
ground_height = 0.1
gravity = 0  # -9.8
penalty = 1e4
damping = 0


def allocate_fields():
    ti.root.dense(ti.l,
                  max_steps).dense(ti.i,
                                   n_objects).place(x, v, rotation, omega,
                                                    v_inc, omega_inc)
    ti.root.dense(ti.i, n_objects).place(halfsize, inverse_mass,
                                         inverse_inertia)
    ti.root.place(loss, friction)
    ti.root.lazy_grad()


dt = 0.0002
learning_rate = 1.0


@ti.func
def rotation_matrix(r):
    return ti.Matrix([[ti.cos(r), -ti.sin(r)], [ti.sin(r), ti.cos(r)]])


@ti.kernel
def initialize_properties():
    for i in range(n_objects):
        inverse_mass[i] = 1.0 / (4 * halfsize[i][0] * halfsize[i][1])
        inverse_inertia[i] = 1.0 / (4 / 3 * halfsize[i][0] * halfsize[i][1] *
                                    (halfsize[i][0] * halfsize[i][0] +
                                     halfsize[i][1] * halfsize[i][1]))
        # ti.print(inverse_mass[i])
        # ti.print(inverse_inertia[i])


@ti.func
def to_world(t, i, rela_x):
    rot = rotation[t, i]
    rot_matrix = rotation_matrix(rot)

    rela_pos = rot_matrix @ rela_x
    rela_v = omega[t, i] * ti.Vector([-rela_pos[1], rela_pos[0]])

    world_x = x[t, i] + rela_pos
    world_v = v[t, i] + rela_v

    return world_x, world_v, rela_pos


@ti.func
def apply_impulse(t, i, impulse, location):
    ti.atomic_add(v_inc[t + 1, i], impulse * inverse_mass[i])
    ti.atomic_add(omega_inc[t + 1, i],
                  (location - x[t, i]).cross(impulse) * inverse_inertia[i])


@ti.kernel
def collide(t: ti.i32):
    for i in range(n_objects):
        hs = halfsize[i]
        for k in ti.static(range(4)):
            f = friction[None]
            # the corner for collision detection
            offset_scale = ti.Vector([k % 2 * 2 - 1, k // 2 % 2 * 2 - 1])

            corner_x, corner_v, rela_pos = to_world(t, i, offset_scale * hs)
            corner_v = corner_v + dt * gravity * ti.Vector([0.0, 1.0])

            # Apply impulse so that there's no sinking
            normal = ti.Vector([0.0, 1.0])
            tao = ti.Vector([1.0, 0.0])

            rn = rela_pos.cross(normal)
            rt = rela_pos.cross(tao)
            impulse_contribution = inverse_mass[i] + (rn) ** 2 * \
                                   inverse_inertia[i]
            timpulse_contribution = inverse_mass[i] + (rt) ** 2 * \
                                    inverse_inertia[i]

            rela_v_ground = normal.dot(corner_v)

            impulse = 0.0
            timpulse = 0.0
            if rela_v_ground < 0 and corner_x[1] < ground_height:
                impulse = -(1 +
                            elasticity) * rela_v_ground / impulse_contribution
                if impulse > 0:
                    # friction
                    timpulse = -corner_v.dot(tao) / timpulse_contribution
                    timpulse = ti.min(f * impulse,
                                      ti.max(-f * impulse, timpulse))

            if corner_x[1] < ground_height:
                # apply penalty
                impulse = impulse - dt * penalty * (
                    corner_x[1] - ground_height) / impulse_contribution

            apply_impulse(t, i, impulse * normal + timpulse * tao, corner_x)


@ti.kernel
def advance(t: ti.i32):
    for i in range(n_objects):
        s = math.exp(-dt * damping)
        v[t, i] = s * v[t - 1, i] + v_inc[t, i] + dt * gravity * ti.Vector(
            [0.0, 1.0])
        x[t, i] = x[t - 1, i] + dt * v[t, i]
        omega[t, i] = s * omega[t - 1, i] + omega_inc[t, i]
        rotation[t, i] = rotation[t - 1, i] + dt * omega[t, i]


@ti.kernel
def compute_loss(t: ti.i32):
    loss[None] = x[t, head_id][0]


gui = ti.GUI("Rigid Body", (1024, 1024), background_color=0xFFFFFF)


def forward(output=None, visualize=True):

    initialize_properties()

    interval = vis_interval
    total_steps = steps
    if output:
        interval = output_vis_interval
        os.makedirs('rigid_body/{}/'.format(output), exist_ok=True)
        total_steps *= 2

    for t in range(1, total_steps):
        collide(t - 1)
        advance(t)

        if (t + 1) % interval == 0 and visualize:
            gui.clear()
            for i in range(n_objects):
                points = []
                for k in range(4):
                    offset_scale = [[-1, -1], [1, -1], [1, 1], [-1, 1]][k]
                    rot = rotation[t, i]
                    rot_matrix = np.array([[math.cos(rot), -math.sin(rot)],
                                           [math.sin(rot),
                                            math.cos(rot)]])

                    pos = np.array([x[t, i][0], x[t, i][1]
                                    ]) + offset_scale * rot_matrix @ np.array(
                                        [halfsize[i][0], halfsize[i][1]])
                    points.append((pos[0], pos[1]))

                for k in range(4):
                    gui.line(points[k], points[(k + 1) % 4], color=0x0, radius=2)

            offset = 0.003
            gui.line((0.05, ground_height - offset),
                     (0.95, ground_height - offset), color=0xAAAAAA, radius=2)

            if output:
                gui.screenshot('rigid_body/{}/{:04d}.png'.format(output, t))
            else:
                gui.show()

    loss[None] = 0
    compute_loss(steps - 1)


@ti.kernel
def clear_states():
    for t in range(0, max_steps):
        for i in range(0, n_objects):
            v_inc[t, i] = ti.Vector([0.0, 0.0])
            omega_inc[t, i] = 0.0


def main():
    allocate_fields()
    for fric in [0, 1]:
        losses = []
        grads = []
        rots = []
        friction[None] = fric
        for i in range(-20, 20):
            x[0, 0] = [0.7, 0.5]
            v[0, 0] = [-1, -2]
            halfsize[0] = [0.1, 0.1]
            rot = (i + 0.5) * 0.001
            rotation[0, 0] = rot
            # forward('initial')
            # for iter in range(50):
            clear_states()

            with ti.Tape(loss):
                forward(visualize=True)

            print('Iter=', i, 'Loss=', loss[None])
            print(omega.grad[0, 0])
            losses.append(loss[None])
            grads.append(omega.grad[0, 0] * 20)
            rots.append(math.degrees(rot))
            # x[0, 0][0] = x[0, 0][0] - x.grad[0, 0][0] * learning_rate
        plt.plot(rots, losses, 'x', label='coeff of friction={}'.format(fric))
    fig = plt.gcf()
    plt.legend()
    fig.set_size_inches(5, 3)
    plt.ylim(0.2, 0.55)
    plt.title('Rigid Body Simulation Discontinuity')
    plt.ylabel('Loss (m)')
    plt.xlabel('Initial Rotation Angle (degrees)')
    plt.tight_layout()
    # plt.plot(grads)
    plt.show()


if __name__ == '__main__':
    main()
