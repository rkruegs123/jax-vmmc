import pdb

import jax.numpy as jnp
from jax import random
from jax import vmap
from jax.config import config as jax_config

from jax_md import util
from jax_md import rigid_body

from functools import partial


FLAGS = jax_config.FLAGS
f32 = util.f32
f64 = util.f64

dtype = f32
if FLAGS.jax_enable_x64:
    dtype = f64


@partial(vmap, in_axes=(0, None))
def rand_quat(key, dtype):
    return rigid_body.random_quaternion(key, dtype)


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


def get_rand_rigid_body(n, box_size, key):
    pos_key, quat_key = random.split(key, 2)

    R = box_size * random.uniform(pos_key, (n, 3), dtype=dtype)
    quat_key = random.split(quat_key, n)
    quaternion = rand_quat(quat_key, dtype)

    body = rigid_body.RigidBody(R, quaternion)
    return body

if __name__ == "__main__":
    pass
