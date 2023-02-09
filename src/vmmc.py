import pdb
from functools import partial
from tqdm import tqdm

import jax
import jax.numpy as jnp
from jax import random
from jax import vmap
from jax.config import config as jax_config
jax_config.update('jax_enable_x64', True)

from jax_md import rigid_body

import utils


"""
Strategy:

Notation:
- A is prelink table
- B is reverse link table (note: we compute separately for now, but could be A.T)
- A * (1 - B) is frustrated
- not A is failed -- note: does not include not B
- A and B is link formed -- note: includes the `min` step

Procedure for precomputing table of links (`all_link_coinflips`):
- compute A, including its coinflips. `prelink_coinflips`
- compute failed matrix, F, as (not A_coinflip)
- compute B as min(1, A / A.T) element-wise (the `min` expression)
- compute `B_coinflip`
- compute C: A_coinflip AND B_coinflip -- this is what you use in the floodfill
- update C: set diagonal to be 1

Procedure for building cluster given table of links -- flood fill:
- given adjacency matrix A: nxn
- randomly sample a seed vertex
- get the corresponding row in the precomputed, one-hot link matrix
  - x: (n,)
- repeat x*A: (n,) n times
  - scan over x
  - final x will just be things in the cluster.

Note:
- assumes that order doesn't matter
  - i.e. link formation of ij doesn't depend on link formation of kl
"""
def vmmc(body, gen_tables_fn, key, n_steps=10, temp=0.3, rot_threshold=0.5):
    n = body.center.shape[0]
    seed_vertices_key, move_key, key = random.split(key, 3)
    seed_vertices = jax.random.randint(seed_vertices_key, (n_steps,), minval=0, maxval=n-1)

    # Pre-select all the moves
    # note: 0 is translational, 1 is rotational
    # note: when doing cluster MC for non-RigidBody stuff, while rotation is not defined for an individual paticle, rotation *is* defined for a cluster. So, we can just cast it to a RigidBody, d oMC wtith translation + rotation and return the center at the end
    # note: rot_threshold can be tuned to change the likelihood of move types
    rotation_thresholds = random.uniform(move_key, shape=(n_steps,))
    move_types = (rotation_thresholds > rot_threshold).astype(jnp.int32)


    beta = 1/temp
    def boltz(x):
        return jnp.exp(-beta * x)
    iter_keys = random.split(key, n_steps)

    mu = body
    traj = [mu]

    identity_quaternion_vec = jnp.array([1.0, 0.0, 0.0, 0.0]) # note: [1.0, 0.0, 0.0, 0.0] is the identity for quat. multiplication
    identity_translation = jnp.array([0.0, 0.0, 0.0]) # note: under addition

    for i, iter_key, seed_vertex, move_type in tqdm(zip(range(n_steps), iter_keys, seed_vertices, move_types)):
        iter_key, move_key = random.split(iter_key, 2)


        trans_move = jnp.where(move_type == 0,
                               utils.gen_random_displacement(r_min=0.5, r_max=1.0, key=move_key),
                               identity_translation)
        # note: can only do a `where` with arrays
        rot_move_vec = jnp.where(move_type == 1,
                                 utils.rand_quat_vec(move_key),
                                 identity_quaternion_vec) # FIXME: don't sample such big rotations! Check HOOMD for how to restrict the range here. TBD.
        rot_move = rigid_body.Quaternion(rot_move_vec)

        nu = rigid_body.RigidBody(mu.center + trans_move, rot_move * mu.orientation)

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
        mu_updated_orientation_vec = jnp.multiply(mu.orientation.vec, jnp.expand_dims(1-cluster, axis=1)) + jnp.multiply(nu.orientation.vec, jnp.expand_dims(cluster, axis=1))
        mu_updated_orientation = rigid_body.Quaternion(mu_updated_orientation_vec)
        mu_updated = rigid_body.RigidBody(mu_updated_center, mu_updated_orientation)


        # check rejection with C * F * (1 - C)
        # is_frustrated = jnp.matmul(jnp.matmul(cluster, frustrated), 1 - cluster)
        is_frustrated = jnp.outer(cluster, 1-cluster) * frustrated
        is_frustrated = is_frustrated.sum()

        # note: terms in `jnp.where` have to be arrays
        mu_center = jnp.where(is_frustrated, mu.center, mu_updated.center)
        mu_orientation = jnp.where(is_frustrated, mu.orientation.vec, mu_updated.orientation.vec)
        mu = rigid_body.RigidBody(center=mu_center,
                                  orientation=rigid_body.Quaternion(mu_orientation))
        traj.append(mu)
    return mu, traj

"""
To compute for probabilities, we need the following:
- proability of type of move (translational vs. orientation) -- 0.5
  - note that in the same way we don't need the probability of the seed vertex, we may not need this
- (i) probability of links in cluster
  - just mask out rows/cols not in the cluster and take the product
  - note: since we really care about log probs, can just take the differences in the boltzmann (with the appropriate min/max)
  - also, have to avoid double counting (just divide by 2)
- (ii) probability of no frustrated links on the boundary
  - recall that `jnp.outer(cluster, 1-cluster)` has entry ij=1 when i is in the cluster and j is not
  - note: probability of not being frustrated is the probability that all boundary links outright failed
  - element-wise multiplication of `jnp.outer(cluster, 1-cluster)` and (1-A), where A is the probability of the prelinks forming
    - note that we can optimize for summing over logs instead of multiplying probailities and then taking the log (by using the diffs that we eventually boltzmann)
- (iii) Probability of 1 or more frustrated links
  - 1 - (ii)
- probability of step:
  - if move accepted: (i) * (ii)
  - if move not accepted: (i) * (iii)

- note: would be nice if we could directly compute the rejection probability, akin to how monte carlo algorithms are tuned to have a particular acceptance rate
- note: could also include a flag just to do single particle
- note: maybe think of some simple test cases
- note: look into references with megan on how and why cluster size is limited to test dynamics
  - note: jamie makes a good point that large clusters will be more subject to being frustrated

"""


if __name__ == "__main__":
    pass


    # note: we still haven't quite worked out how to do max cluster size
    # option 1: only do d iterations. problem is that graph size doesn't correspond to space
    # option 2: do the whole thing then only sample d things.
    # problems with all of these: maintaining connectedness, uniformly sampling something of size `d`
    # Also note: are we only cecking furstareted for the boundary? does it matter?
