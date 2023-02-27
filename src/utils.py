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

def rand_2d_rotation(mu, a_max, theta_max, seed_vertex, key):
    n = mu.center.shape[0]
    a_key, theta_key, u_key = random.split(key, 3)

    u_theta = random.uniform(u_key, minval=0, maxval=2*jnp.pi)
    u = jnp.array([jnp.cos(u_theta), jnp.sin(u_theta)])
    a = random.uniform(a_key, minval=0, maxval=a_max)
    rot_center = mu[seed_vertex].center + a*u

    theta = random.uniform(theta_key, minval=-theta_max, maxval=theta_max)
    rot_matrix = jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])

    nu_center = (rot_matrix @ (mu.center - rot_center).T).T + rot_center
    nu = rigid_body.RigidBody(nu_center, mu.orientation + theta)
    return nu


# Copied from: https://rowan.readthedocs.io/en/latest/package-rowan.html#rowan.from_axis_angle
def from_axis_angle(axis, angle):
    # First reshape angles and compute the half angle
    ha = angle / 2.0

    # Normalize the vector
    norm = jnp.linalg.norm(axis)
    u = axis / norm

    # Compute the components of the quaternions
    scalar_vec = jnp.array([jnp.cos(ha)])
    vec_comp = jnp.sin(ha) * u

    return jnp.concatenate((scalar_vec, vec_comp))


def rand_3d_rotation(mu, seed_vertex, theta_max, a_max, key):
    n = mu.center.shape[0]
    a_key, u_key, uo_key, theta_key = random.split(key, 4)

    # Get the point around which we will rotate
    rand_vec = random.normal(u_key, shape=(3,))
    u = rand_vec / jnp.linalg.norm(rand_vec)
    a = random.uniform(a_key, minval=0, maxval=a_max)
    rot_center = mu[seed_vertex].center + a*u

    # Get a random axis and angle, and the corresponding quaternion
    unnormed_rot_axis = random.normal(uo_key, shape=(3,))
    rot_axis = unnormed_rot_axis / jnp.linalg.norm(unnormed_rot_axis)
    theta = random.uniform(theta_key, minval=-theta_max, maxval=theta_max)
    quat_vec = from_axis_angle(rot_axis, theta)
    quat = rigid_body.Quaternion(quat_vec)

    # Calculate nu w.r.t. the center of rotation and the quaternion
    mu_center_adj = mu.center - rot_center
    nu_center_adj = rigid_body.quaternion_rotate(quat, mu_center_adj)
    nu_center = nu_center_adj + rot_center
    nu_orientation = mu.orientation * quat

    return rigid_body.RigidBody(nu_center, nu_orientation)

def rand_3d_translation(mu, r_min, r_max, key):
    disp = gen_random_displacement(r_min=r_min, r_max=r_max, key=key)
    return rigid_body.RigidBody(mu.center+disp, mu.orientation)




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
