import pdb
from functools import partial

import jax
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

import utils




def gen_tables(mu, nu, pairwise_energies_fn):
    eps_mu_mu = pairwise_energies_fn(mu, mu)
    eps_mu_nu = pairwise_energies_fn(mu, nu)
    eps_nu_mu = pairwise_energies_fn(nu, mu)

    return eps_mu_mu, eps_mu_nu, eps_nu_mu


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


if __name__ == "__main__":
    import vmmc

    n = 10
    # box_size = quantity.box_size_at_number_density(n, 0.1, 3)
    box_size = 5
    displacement, shift = space.periodic(box_size)

    key = random.PRNGKey(0)

    key, body_key = random.split(key, 2)
    init_body = utils.get_rand_rigid_body(n, box_size, body_key)

    fin_state, traj = vmmc.vmmc(init_body,
                                partial(gen_tables, pairwise_energies_fn=get_pairwise_energies),
                                key)
