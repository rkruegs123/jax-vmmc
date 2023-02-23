import pdb

import jax.numpy as jnp
from jax import random
from jax import vmap
from jax.config import config as jax_config
jax_config.update('jax_enable_x64', True)

from jax_md import util
from jax_md import rigid_body

from functools import partial


FLAGS = jax_config.FLAGS
f32 = util.f32
f64 = util.f64

dtype = f32
if FLAGS.jax_enable_x64:
    dtype = f64


# Note: since we vmap, this takes in an array of keys
@partial(vmap, in_axes=(0, None))
def rand_quat(key, dtype):
    return rigid_body.random_quaternion(key, dtype)

# Note: we don't vmap this so it takes in a single key
def rand_quat_vec(key):
    return rigid_body._random_quaternion(key, f64)


# via spherical coordinates. r_min and r_max are the min and max radii of the sphere
# TODO: could sample N from the beginning where N is the number of steps?
def gen_random_displacement(r_min, r_max, key):
    radii_key, theta_key, phi_key = random.split(key, 3)

    radius = random.uniform(radii_key, minval=r_min, maxval=r_max)
    theta = random.uniform(theta_key, minval=0, maxval=jnp.pi)
    phi = random.uniform(phi_key, minval=0, maxval=2*jnp.pi)

    x = radius * jnp.sin(theta) * jnp.cos(phi)
    y = radius * jnp.sin(theta) * jnp.sin(phi)
    z = radius * jnp.cos(theta)

    return jnp.array([x, y, z])


def gen_random_displacement_2d(r_min, r_max, key):
    radii_key, theta_key = random.split(key, 2)

    radius = random.uniform(radii_key, minval=r_min, maxval=r_max)
    theta = random.uniform(theta_key, minval=0, maxval=2*jnp.pi)

    x = radius * jnp.cos(theta)
    y = radius * jnp.sin(theta)

    return jnp.array([x, y])


def rand_2d_translation(mu, r_min, r_max, key):
    trans_move = gen_random_displacement_2d(r_min, r_max, key)
    nu = rigid_body.RigidBody(mu.center + trans_move, mu.orientation)
    return nu

def rand_2d_rotation(mu, a_max, theta_max, key):
    n = mu.center.shape[0]
    a_key, theta_key, center_key, u_key = random.split(key, 4)

    center_idx = random.randint(center_key, shape=(), minval=0, maxval=n-1)
    u_theta = random.uniform(u_key, minval=0, maxval=2*jnp.pi)
    u = jnp.array([jnp.cos(u_theta), jnp.sin(u_theta)])
    a = random.uniform(a_key, minval=0, maxval=a_max)
    rot_center = mu[center_idx].center + a*u

    theta = random.uniform(theta_key, minval=-theta_max, maxval=theta_max)
    rot_matrix = jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])

    nu_center = (rot_matrix @ (mu.center - rot_center).T).T + rot_center
    nu = rigid_body.RigidBody(nu_center, mu.orientation + theta)
    return nu



def get_rand_rigid_body(n, box_size, key):
    pos_key, quat_key = random.split(key, 2)

    R = box_size * random.uniform(pos_key, (n, 3), dtype=dtype)
    quat_key = random.split(quat_key, n)
    quaternion = rand_quat(quat_key, dtype)

    body = rigid_body.RigidBody(R, quaternion)
    return body


def get_rand_rigid_body_2d(n, box_size, key):
    pos_key, angle_key = random.split(key, 2)

    R = box_size * random.uniform(pos_key, (n, 2), dtype=dtype)
    angle = random.uniform(angle_key, (n,), dtype=dtype) * jnp.pi * 2

    body = rigid_body.RigidBody(R, angle)
    return body

if __name__ == "__main__":
    pass
