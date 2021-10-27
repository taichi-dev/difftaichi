import taichi as ti
import numpy as np
import math
import os
from imageio import imwrite

os.makedirs('output_volume_renderer', exist_ok=True)

real = ti.f32
ti.init(default_fp=real, arch=ti.cuda)

num_iterations = 100
res = 512
density_res = 128
inv_density_res = 1.0 / density_res
res_f32 = float(res)
dx = 0.02
n_views = 7
torus_r1 = 0.4
torus_r2 = 0.1
fov = 1
camera_origin_radius = 1
marching_steps = 1000
learning_rate = 15

scalar = lambda: ti.field(dtype=real)

density = scalar()
target_images = scalar()
images = scalar()
loss = scalar()

ti.root.dense(ti.ijk, density_res).place(density)
ti.root.dense(ti.i, n_views).dense(ti.jk, res).place(target_images, images)
ti.root.place(loss)
ti.root.lazy_grad()


@ti.func
def in_box(x, y, z):
    # The density grid is contained in a unit box [-0.5, 0.5] x [-0.5, 0.5] x [-0.5, 0.5]
    return x >= -0.5 and x < 0.5 and y >= -0.5 and y < 0.5 and z >= -0.5 and z < 0.5


@ti.kernel
def ray_march(field: ti.template(), angle: ti.f32, view_id: ti.i32):
    for pixel in range(res * res):
        for k in range(marching_steps):
            x = pixel // res
            y = pixel - x * res

            camera_origin = ti.Vector([
                camera_origin_radius * ti.sin(angle), 0,
                camera_origin_radius * ti.cos(angle)
            ])
            dir = ti.Vector([
                fov * (ti.cast(x, ti.f32) /
                       (res_f32 / 2.0) - res_f32 / res_f32),
                fov * (ti.cast(y, ti.f32) / (res_f32 / 2.0) - 1.0), -1.0
            ])

            length = ti.sqrt(dir[0] * dir[0] + dir[1] * dir[1] +
                             dir[2] * dir[2])
            dir /= length

            rotated_x = dir[0] * ti.cos(angle) + dir[2] * ti.sin(angle)
            rotated_z = -dir[0] * ti.sin(angle) + dir[2] * ti.cos(angle)
            dir[0] = rotated_x
            dir[2] = rotated_z
            point = camera_origin + (k + 1) * dx * dir

            # Convert to coordinates of the density grid box
            box_x = point[0] + 0.5
            box_y = point[1] + 0.5
            box_z = point[2] + 0.5

            # Density grid location
            index_x = ti.cast(ti.floor(box_x * density_res), ti.i32)
            index_y = ti.cast(ti.floor(box_y * density_res), ti.i32)
            index_z = ti.cast(ti.floor(box_z * density_res), ti.i32)
            index_x = ti.max(0, ti.min(index_x, density_res - 1))
            index_y = ti.max(0, ti.min(index_y, density_res - 1))
            index_z = ti.max(0, ti.min(index_z, density_res - 1))

            flag = 0
            if in_box(point[0], point[1], point[2]):
                flag = 1

            contribution = density[index_z, index_y, index_x] * flag

            field[view_id, y, x] += contribution


@ti.kernel
def compute_loss(view_id: ti.i32):
    for i in range(res):
        for j in range(res):
            loss[None] += (images[view_id, i, j] -
                           target_images[view_id, i, j])**2 * (1.0 /
                                                               (res * res))


@ti.kernel
def clear_images():
    for v, i, j in images:
        images[v, i, j] = 0


@ti.kernel
def clear_density():
    for i, j, k in density:
        density[i, j, k] = 0
        density.grad[i, j, k] = 0


def create_target_images():
    for view in range(n_views):
        ray_march(target_images, math.pi / n_views * view - math.pi / 2.0,
                  view)

        img = np.zeros((res, res), dtype=np.float32)
        for i in range(res):
            for j in range(res):
                img[i, j] = target_images[view, i, j]
        img /= np.max(img)
        img = 1 - img

        imwrite("{}/target_{:04d}.png".format("output_volume_renderer", view),
                100 * img)


@ti.func
def in_torus(x, y, z):
    len_xz = ti.sqrt(x * x + z * z)
    qx = len_xz - torus_r1
    len_q = ti.sqrt(qx * qx + y * y)
    dist = len_q - torus_r2
    return dist < 0


@ti.kernel
def create_torus_density():
    for i, j, k in density:
        # Convert to density coordinates
        x = ti.cast(k, ti.f32) * inv_density_res - 0.5
        y = ti.cast(j, ti.f32) * inv_density_res - 0.5
        z = ti.cast(i, ti.f32) * inv_density_res - 0.5

        # Swap x, y to rotate the torus
        if in_torus(y, x, z):
            density[i, j, k] = inv_density_res
        else:
            density[i, j, k] = 0.0


@ti.kernel
def apply_grad():
    # gradient descent
    for i, j, k in density:
        density[i, j, k] -= learning_rate * density.grad[i, j, k]
        density[i, j, k] = ti.max(density[i, j, k], 0)


def main():
    if not os.path.exists('bunny_128.bin'):
        print(
            '\n***\nPlease download bunny_128.bin and put in the current working directory. URL: https://github.com/yuanming-hu/taichi_assets/releases/download/llvm8/bunny_128.bin '
        )
        exit(0)
    volume = np.fromfile("bunny_128.bin", dtype=np.float32).reshape(
        (density_res, density_res, density_res))
    for i in range(density_res):
        for j in range(density_res):
            for k in range(density_res):
                density[i, j, k] = volume[i, density_res - j - 1, k]

    #create_torus_density()
    create_target_images()
    clear_density()

    for iter in range(num_iterations):
        clear_images()
        with ti.Tape(loss):
            for view in range(n_views):
                ray_march(images, math.pi / n_views * view - math.pi / 2.0,
                          view)
                compute_loss(view)

        views = images.to_numpy()
        for view in range(n_views):
            img = views[view]
            m = np.max(img)
            if m > 0:
                img /= m
            img = 1 - img
            imwrite(
                "{}/image_{:04d}_{:04d}.png".format("output_volume_renderer",
                                                    iter, view),
                (255 * img).astype(np.uint8))

        print('Iter', iter, ' Loss =', loss[None])
        apply_grad()


if __name__ == '__main__':
    main()
