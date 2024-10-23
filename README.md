# Code for reproducing results from `Tracer Budgets on Lagrangian Trajector`

The majority of the codes that produced the result in the case studies are hosted in the python package [seaduck](https://github.com/MaceKuailv/seaduck). 

The `dic` folder contains how to modify the MITgcm tutorial BGC run to get the necessary diagnostics, how to formulate the Eulerian budget, and how to simulate the particles with phosphate and dissolved inorganic carbon budget. 

The `heat` folder contains how to create the anomaly budget from daily mean ECCO, which can be accessed via [ECCO home page](https://ecco-group.org/) or [sciserver](https://apps.sciserver.org). The backward simulation and the budget calculation is also included. 

The aesthetics and plotting is done in the folder `james_plot`.
