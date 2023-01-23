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

"""
Note:
- assumes that order doesn't matter
  - i.e. link formation of ij doesn't depend on link formation of kl
"""
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
        # return energy.soft_sphere(dist)
        # return energy.simple_spring(dist)
        # return energy.lennard_jones(dist)
        return energy.simple_spring(dist, length=0.5) + energy.soft_sphere(dist)


    get_pairwise_energies = vmap(vmap(pairwise_energy_fn, in_axes=(None, 0)),
                                 in_axes=(0, None))

    eps_mu_mu, eps_mu_nu, eps_nu_mu = gen_tables(mu, nu, get_pairwise_energies)

    key = random.PRNGKey(0) # note: fixed for now

    prelink_diffs = eps_nu_mu - eps_mu_mu # eps_ip_j^mu - eps_i_j^mu
    temp = 300
    beta = 1/temp
    def boltz(x):
        return jnp.exp(-beta * x)
    prelink_boltz = boltz(prelink_diffs)
    prelink_probs = jnp.maximum(0, 1-prelink_boltz) # can be made branchless
    prelink_coinflip_thresholds = random.uniform(key, shape=prelink_probs.shape)
    prelink_coinflips = (prelink_probs > prelink_coinflip_thresholds).astype(jnp.int32)

    rev_link_diffs = eps_mu_nu - eps_mu_mu # eps_i_jp^mu - eps_i_j^mu
    rev_link_boltz = boltz(rev_link_diffs)
    rev_link_probs = jnp.maximum(0, 1-rev_link_boltz) # can be made branchless
    # rev_link_coinflip_thresholds = random.uniform(key, shape=rev_link_probs.shape)
    # rev_link_coinflips = (rev_link_probs > rev_link_coinflip_thresholds).astype(jnp.int32)

    ratio = prelink_probs / rev_link_probs
    uncorrected_probs = jnp.minimum(ratio, 1.0) # can be made branchless

    all_link_probs = jnp.multiply(prelink_coinflips, uncorrected_probs)
    all_link_coinflip_thresholds = random.uniform(key, shape=all_link_probs.shape)
    all_link_coinflips = (all_link_probs > all_link_coinflip_thresholds).astype(jnp.int32)

    pdb.set_trace()

    print("done")



    """
    Flood fill:
    - given adjacency matrix A: nxn
    - randomly sample a seed vertex
    - get the corresponding row in the precomputed, one-hot link matrix
      - x: (n,)
    - repeat x*A: (n,) n times
      - scan over x
      - final x will just be things in the cluster.
    """

    # also, do the frustrated thing

    # for next time: flood fill and frustrated thing. maybe a simple test case as well.

    # note: we still haven't quite worked out how to do max cluster size
    # option 1: only do d iterations. problem is that graph size doesn't correspond to space
    # option 2: do the whole thing then only sample d things.
    # problems with all of these: maintaining connectedness, uniformly sampling something of size `d`
