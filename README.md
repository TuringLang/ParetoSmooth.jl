# JuLOOa

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://invenia.github.io/PkgTemplates.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://invenia.github.io/PkgTemplates.jl/dev)
[![CI](https://github.com/invenia/PkgTemplates.jl/workflows/CI/badge.svg)](https://github.com/invenia/PkgTemplates.jl/actions?query=workflow%3ACI)
[![Codecov](https://codecov.io/gh/invenia/PkgTemplates.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/invenia/PkgTemplates.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

**An implementation of Pareto Smoothed Importance Sampling in Julia.**

## Installation

Install with Pkg, just like any other registered Julia package:

```jl
pkg> add PkgTemplates  # Press ']' to enter the Pkg REPL mode.
```

## Usage

### Interactive Generation

Use the `psis` command to perform Pareto Smoothed Importance Sampling, and return a PSIS object for further analysis.

