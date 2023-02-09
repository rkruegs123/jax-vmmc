# jax-vmmc

An implementation of Virtual Move Monte Carlo in JAX

Misc notes:
- https://github.com/glotzerlab/freud/blob/master/cpp/cluster/Cluster.cc
- note sampling of orientation and translation
- papers
  - following: https://journals.aps.org/pre/abstract/10.1103/PhysRevE.89.033307
  - original papers:
    - https://pubs.rsc.org/en/content/articlelanding/2009/SM/B810031D
    - https://aip.scitation.org/doi/10.1063/1.2790421



## Feb 9, 2023
- we are doing the rotations wrong in both 2d and 3d. need to rotate about a randomly sampled axis of rotation
  - and/or center of rotation
  - see page 8 of the Ruzicka and Allen ("the paper")
- minimum translation should always be 0
  - might allow that cluster in the morse example to separate
