import pdb
from functools import partial
from tqdm import tqdm

import jax
import jax.numpy as jnp
from jax import random
from jax import vmap, jit
from jax.config import config as jax_config
jax_config.update('jax_enable_x64', True)

from jax_md import rigid_body

import utils



def vmmc(body, gen_tables_fn, key, n_steps=10, temp=0.3, rot_threshold=0.5):
    n = body.center.shape[0]
    seed_vertices_key, move_key, key = random.split(key, 3)
    seed_vertices = jax.random.randint(seed_vertices_key, (n_steps,), minval=0, maxval=n-1)

    rotation_thresholds = random.uniform(move_key, shape=(n_steps,))
    move_types = (rotation_thresholds > rot_threshold).astype(jnp.int32)


    beta = 1/temp
    def boltz(x):
        return jnp.exp(-beta * x)
    iter_keys = random.split(key, n_steps)

    mu = body
    traj = [mu]

    identity_theta = 0.0
    identity_translation = jnp.array([0.0, 0.0]) # note: under addition

    @jit
    def step_fn(mu, iter_key, seed_vertex, move_type):
        iter_key, move_key = random.split(iter_key, 2)

        trans_move = jnp.where(move_type == 0,
                               utils.gen_random_displacement_2d(r_min=0.5, r_max=1.0, key=move_key),
                               identity_translation)
        rot_move = jnp.where(move_type == 1,
                             random.uniform(move_key, minval=0.0, maxval=jnp.pi / 6),
                             identity_theta)

        nu = rigid_body.RigidBody(mu.center + trans_move, mu.orientation + rot_move)

        eps_mu_mu, eps_mu_nu, eps_nu_mu = gen_tables_fn(mu, nu)


        # Precompute the coinflips
        prelink_diffs = eps_nu_mu - eps_mu_mu # eps_ip_j^mu - eps_i_j^mu
        prelink_boltz = boltz(prelink_diffs)
        prelink_probs = jnp.maximum(0, 1-prelink_boltz) # can be made branchless
        iter_key, prelink_key = random.split(iter_key, 2)
        prelink_coinflip_thresholds = random.uniform(prelink_key, shape=prelink_probs.shape)
        prelink_coinflips = (prelink_probs > prelink_coinflip_thresholds).astype(jnp.int32) # matrix A

        rev_link_diffs = eps_mu_nu - eps_mu_mu # eps_i_jp^mu - eps_i_j^mu
        rev_link_boltz = boltz(rev_link_diffs)
        rev_link_probs = jnp.maximum(0, 1-rev_link_boltz) # can be made branchless
        ratio = jnp.where(prelink_probs == 0, 0.0, rev_link_probs / prelink_probs)
        all_link_probs = jnp.minimum(ratio, 1.0) # can be made branchless
        iter_key, all_link_key = random.split(iter_key, 2)
        all_link_coinflip_thresholds = random.uniform(all_link_key, shape=all_link_probs.shape)
        arr_B = (all_link_probs > all_link_coinflip_thresholds).astype(jnp.int32) # matrix B

        all_link_coinflips = jnp.multiply(prelink_coinflips, arr_B)
        all_link_coinflips = all_link_coinflips + jnp.eye(all_link_coinflips.shape[0]) # matrix C

        frustrated = jnp.multiply(prelink_coinflips, 1 - arr_B) # frustrated


        # Do the floodfill
        seed_row = all_link_coinflips[seed_vertex]
        def flood_fill_step(curr_cluster, v_idx):
            # return curr_cluster + jnp.matmul(curr_cluster, all_link_coinflips), None # note: only have to add because we don't add the identity to all_link_coinflips
            return jnp.matmul(curr_cluster, all_link_coinflips), None
        cluster, _ = jax.lax.scan(flood_fill_step, seed_row, jnp.arange(n))
        cluster = jnp.minimum(cluster, 1)


        # to get `mu_updated`, we treat cluster like a mask
        mu_updated_center = jnp.multiply(mu.center, jnp.expand_dims(1-cluster, axis=1)) + jnp.multiply(nu.center, jnp.expand_dims(cluster, axis=1))
        mu_updated_orientation = jnp.multiply(mu.orientation, jnp.expand_dims(1-cluster, axis=1)) + jnp.multiply(nu.orientation, jnp.expand_dims(cluster, axis=1))
        mu_updated = rigid_body.RigidBody(mu_updated_center, mu_updated_orientation)


        # check rejection with C * F * (1 - C)
        # is_frustrated = jnp.matmul(jnp.matmul(cluster, frustrated), 1 - cluster)
        is_frustrated = jnp.outer(cluster, 1-cluster) * frustrated
        is_frustrated = is_frustrated.sum()

        # note: terms in `jnp.where` have to be arrays
        mu_center = jnp.where(is_frustrated, mu.center, mu_updated.center)
        mu_orientation = jnp.where(is_frustrated, mu.orientation, mu_updated.orientation)
        return rigid_body.RigidBody(center=mu_center,
                                    orientation=mu_orientation)


    for i, iter_key, seed_vertex, move_type in tqdm(zip(range(n_steps), iter_keys, seed_vertices, move_types)):
        mu = step_fn(mu, iter_key, seed_vertex, move_type)
        traj.append(mu)
    return mu, traj


if __name__ == "__main__":
    pass
