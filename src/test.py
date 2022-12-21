import pdb

import jax.numpy as jnp
from jax import random
from jax import vmap
from jax.config import config as jax_config

from jax_md import quantity
from jax_md import simulate
from jax_md import space
from jax_md import energy
from jax_md import test_util
from jax_md import partition
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

def gen_tables(mu, nu, pairwise_energies_fn):
    eps_mu_mu = pairwise_energies_fn(mu, mu)
    eps_mu_nu = pairwise_energies_fn(mu, nu)
    eps_nu_mu = pairwise_energies_fn(nu, mu)

    return eps_mu_mu, eps_mu_nu, eps_nu_mu


if __name__ == "__main__":
    n = 50
    box_size = quantity.box_size_at_number_density(n, 0.1, 3)
    displacement, shift = space.periodic(box_size)

    key = random.PRNGKey(0)

    key, body_key, displacement_key = random.split(key, 3)
    mu = get_rand_rigid_body(n, box_size, body_key)
    move = gen_random_displacement(r_min=0.5, r_max=1.0, key=displacement_key)
    nu = rigid_body.RigidBody(mu.center + move, mu.orientation)

    # center_energy_fn = energy.soft_sphere(displacement)

    # p1 and p2 are rigid bodies of size 1
    def pairwise_energy_fn(p1, p2):
        dist = jnp.linalg.norm(p1.center - p2.center)
        # return center_energy_fn(dist)
        return energy.soft_sphere(dist)


    get_pairwise_energies = vmap(vmap(pairwise_energy_fn, in_axes=(None, 0)),
                                 in_axes=(0, None))

    t1, t2, t3 = gen_tables(mu, nu, get_pairwise_energies)

    print("done")
