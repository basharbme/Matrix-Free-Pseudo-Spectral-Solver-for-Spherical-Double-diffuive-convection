# Matrix-Free-Pseudo-Spectral-Solver-for-Spherical-Double-diffuive-convection

Spatial localisation typically arises in large aspect ratio domains. To study this phenomenon requires a high dimensional 
system N ~10^6 unknowns. This makes the construction and storage of matrices impractical on small workstations. By making 
use of iterative Krylov-subspace methods to avoid the construction of large matrices and pseudo spectral collocation methods
to minimise computational complexity ~ N*log(N), this code efficiently preforms:

1) time-stepping using a Crank-Nicolson or Euler Implicit schceme,
2) Netwon Iteration & paramater continuation using GMRES or Bi-cgstab routines, 
3) Pseudo-Arc length paramater continuation using GMRES or Bi-cgstab routines, 
4) Stability analysis using Arnoldi iteration.
