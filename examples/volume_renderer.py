import taichi as ti
import numpy as np
import math
import os
from imageio import imwrite

os.makedirs('output_volume_renderer', exist_ok=True)

ti_float = ti.f32
ti.init(default_fp=ti_float, arch=ti.cuda)

num_iterations = 100
render_resolution = 512
volume_dimension = 128
inv_vol_dim = 1.0 / volume_dimension
render_res_f32 = float(render_resolution)
dx = 0.02
n_views = 7
fov = 1
camera_origin_radius = 1
marching_steps = 1000
learning_rate = 15


def create_scalar_field():
    return ti.field(dtype=ti_float)


density = create_scalar_field()
target_images = create_scalar_field()
images = create_scalar_field()
loss = create_scalar_field()

# Configuring data layouts here
# see https://taichi.graphics/docs/develop/documentation/advanced/layout.html
ti.root.dense(ti.ijk, volume_dimension).place(density)
ti.root.dense(ti.l,
              n_views).dense(ti.ij,
                             render_resolution).place(target_images, images)
ti.root.place(loss)
ti.root.lazy_grad()


@ti.func
def in_box(x, y, z):
    # The density grid is contained in a unit box [-0.5, 0.5] x [-0.5, 0.5] x [-0.5, 0.5]
    return -0.5 <= x < 0.5 and -0.5 <= y < 0.5 and -0.5 <= z < 0.5


@ti.kernel
def ray_march(field: ti.template(), angle: ti.f32, view_id: ti.i32):
    for pixel in range(render_resolution * render_resolution):
        for k in range(marching_steps):
            x = pixel // render_resolution
            y = pixel - x * render_resolution
            # camera on z-x plane
            camera_origin = ti.Vector([
                camera_origin_radius * ti.sin(angle), 0,
                camera_origin_radius * ti.cos(angle)
            ])
            direction = ti.Vector([
                fov *
                (ti.cast(x, ti.f32) /
                 (render_res_f32 / 2.0) - render_res_f32 / render_res_f32),
                fov * (ti.cast(y, ti.f32) / (render_res_f32 / 2.0) - 1.0), -1.0
            ])

            length = ti.sqrt(direction[0] * direction[0] +
                             direction[1] * direction[1] +
                             direction[2] * direction[2])
            direction /= length

            # rotate a ray's direction
            rotated_x = direction[0] * ti.cos(angle) + direction[2] * ti.sin(
                angle)
            rotated_z = -direction[0] * ti.sin(angle) + direction[2] * ti.cos(
                angle)
            direction[0] = rotated_x
            direction[2] = rotated_z
            point = camera_origin + (k + 1) * dx * direction

            # Convert to coordinates of the density grid box
            box_x = point[0] + 0.5
            box_y = point[1] + 0.5
            box_z = point[2] + 0.5

            # Density grid location
            index_x = ti.cast(ti.floor(box_x * volume_dimension), ti.i32)
            index_y = ti.cast(ti.floor(box_y * volume_dimension), ti.i32)
            index_z = ti.cast(ti.floor(box_z * volume_dimension), ti.i32)
            # in range [0, vol_dimension -1] ^ 3
            index_x = ti.max(0, ti.min(index_x, volume_dimension - 1))
            index_y = ti.max(0, ti.min(index_y, volume_dimension - 1))
            index_z = ti.max(0, ti.min(index_z, volume_dimension - 1))

            flag = 0
            if in_box(point[0], point[1], point[2]):
                flag = 1

            contribution = density[index_z, index_y, index_x] * flag

            field[view_id, y, x] += contribution


@ti.kernel
def compute_mse(view_id: ti.i32):
    for i in range(render_resolution):
        for j in range(render_resolution):
            loss[None] += (images[view_id, i, j] - target_images[view_id, i, j]
                           )**2 * (1.0 /
                                   (render_resolution * render_resolution))


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

        img = np.zeros((render_resolution, render_resolution),
                       dtype=np.float32)
        for i in range(render_resolution):
            for j in range(render_resolution):
                img[i, j] = target_images[view, i, j]
        img /= np.max(img)
        img = 1 - img

        imwrite("{}/target_{:04d}.png".format("output_volume_renderer", view),
                100 * img)


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
        (volume_dimension, volume_dimension, volume_dimension))
    for i in range(volume_dimension):
        for j in range(volume_dimension):
            for k in range(volume_dimension):
                density[i, j, k] = volume[i, volume_dimension - j - 1, k]

    create_target_images()
    clear_density()

    for iteration in range(num_iterations):
        clear_images()
        with ti.Tape(loss):
            for view in range(n_views):
                ray_march(images, math.pi / n_views * view - math.pi / 2.0,
                          view)
                compute_mse(view)

        views = images.to_numpy()
        for view in range(n_views):
            img = views[view]
            m = np.max(img)
            if m > 0:
                img /= m
            img = 1 - img  # high density gives darker color
            imwrite(
                "{}/image_{:04d}_{:04d}.png".format("output_volume_renderer",
                                                    iteration, view),
                (255 * img).astype(np.uint8))

        print('Iter', iteration, ' Loss =', loss[None])
        apply_grad()


if __name__ == '__main__':
    main()
