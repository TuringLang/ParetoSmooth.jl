# ParetoSmooth

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://TuringLang.github.io/ParetoSmooth.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://TuringLang.github.io/ParetoSmooth.jl/dev)
[![Build Status](https://github.com/TuringLang/ParetoSmooth.jl/workflows/CI/badge.svg)](https://github.com/TuringLang/ParetoSmooth.jl/actions)
[![Coverage](https://codecov.io/gh/TuringLang/ParetoSmooth.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/TuringLang/ParetoSmooth.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)


ParetoSmooth.jl is a Julia package for efficient approximate leave-one-out cross-validation for fitted Bayesian models. We compute LOO-CV using Pareto smoothed importance sampling (PSIS), a modification of importance sampling. More details can be found in Vehtari, Gelman, and Gabry (2017).


If you use this library, please remember to cite both:
```
@misc{ParetoSmooth.jl,
	author  = {Carlos Parada <paradac@carleton.edu>},
	title   = {ParetoSmooth.jl},
	url     = {https://github.com/TuringLang/ParetoSmooth.jl},
	version = {v0.7.1},
	year    = {2021},
	month   = {6}
}
```
and:
```
﻿@Article{Vehtari2017,
  author={Vehtari, Aki
  and Gelman, Andrew
  and Gabry, Jonah},
  title={Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC},
  journal={Statistics and Computing},
  year={2017},
  month={Sep},
  day={01},
  volume={27},
  number={5},
  pages={1413-1432},
  issn={1573-1375},
  doi={10.1007/s11222-016-9696-4},
  url={https://doi.org/10.1007/s11222-016-9696-4}
}
```

