# Pi-Girsanov
Authors: Mingyuan Zhang, Yong Wang, Bettina G. Keller, Hao Wu

This is the official repository for the paper: "π-Girsanov: A Generalized Method to Construct Markov State Models from Non-Equilibrium and Multiensemble Biased Simulations". This paper is currently under review. Preprint is available at: https://arxiv.org/abs/2603.21890.

To setup the environment required to run all these simulation/analysis notebooks used in this study, please run:
```
conda create -n girsanov-torch -c conda-forge -c omnia -c numba -c nvidia -c pytorch pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=11.8 openmm-torch openmm mdtraj netcdf4 mpiplus pymbar numba matplotlib deeptime openmm-plumed numpy scipy scikit-learn python=3.9 mdanalysis pandas
```

Also remember to install a customized version of `openmmtools` for computing the path reweighting factor $M(\omega)$ on-the-fly [here](https://github.com/bkellerlab/reweightingtools?tab=readme-ov-file).
