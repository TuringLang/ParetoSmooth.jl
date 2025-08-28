# ParetoSmooth.jl Development Guide

ParetoSmooth.jl is a Julia package for efficient approximate leave-one-out cross-validation for fitted Bayesian models using Pareto smoothed importance sampling (PSIS). This package integrates with Turing.jl and MCMCChains.jl for Bayesian modeling workflows.

**Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.**

## Working Effectively

### Environment Setup and Package Installation
- Ensure Julia 1.6+ is available: `julia --version`
- Navigate to repository root: `cd /path/to/ParetoSmooth.jl`
- Install and instantiate package dependencies:
  ```bash
  julia --project=. -e "using Pkg; Pkg.instantiate()"
  ```
  - **NEVER CANCEL**: Takes ~35 seconds with precompilation. ALWAYS wait for completion.
  - Precompiles 50+ dependencies including PrettyTables, Distributions, AxisKeys, etc.

### Building and Testing
- **NEVER CANCEL**: Test suite takes **2 minutes** to complete. Set timeout to 5+ minutes.
- Run full test suite:
  ```bash
  julia --project=. -e "using Pkg; Pkg.test()"
  ```
  - Includes comprehensive validation against R reference implementations
  - Tests PSIS algorithms, LOO-CV calculations, and Turing.jl integration
  - Contains 48 test cases across BasicTests, TuringTests, and ComparisonTests

### Documentation Building
- Setup documentation environment:
  ```bash
  julia --project=docs -e "using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()"
  ```
- Build documentation:
  ```bash
  julia --project=docs --color=yes docs/make.jl
  ```
  - **NEVER CANCEL**: Takes ~15 seconds including dependency installation
  - Generates HTML documentation in `docs/build/`

## Validation

### Manual Testing Scenarios
- **ALWAYS** test core PSIS functionality after making changes:
  ```julia
  using ParetoSmooth, Random
  Random.seed!(123)
  log_lik = rand(100, 200, 2)  # 100 data points, 200 samples, 2 chains
  psis_result = psis(log_lik)
  loo_result = psis_loo(log_lik)
  println("PSIS-LOO completed successfully")
  ```

- **ALWAYS** validate Turing.jl integration when modifying extensions:
  ```julia
  # Test requires DynamicPPL and MCMCChains to be available
  # Refer to docs/src/turing.md for complete examples
  ```

### Testing Requirements
- Run tests before committing: `julia --project=. -e "using Pkg; Pkg.test()"`
- **CRITICAL**: NEVER CANCEL tests even if they appear to hang - they take 2 minutes
- All 48 tests must pass for a valid build
- Tests validate against R reference data in `test/data/`

## Common Tasks

### Package Structure
```
ParetoSmooth.jl/
├── src/                    # Core implementation
│   ├── ParetoSmooth.jl    # Main module file
│   ├── ImportanceSampling.jl
│   ├── LeaveOneOut.jl     # PSIS-LOO implementation
│   ├── ModelComparison.jl # Model comparison utilities
│   └── ...
├── ext/                    # Package extensions
│   ├── ParetoSmoothDynamicPPLExt.jl
│   └── ParetoSmoothMCMCChainsExt.jl
├── test/
│   ├── runtests.jl        # Main test runner
│   ├── tests/             # Test suites
│   └── data/              # R reference data (.RData files)
├── docs/                   # Documentation
├── Project.toml           # Dependencies and metadata
└── .github/workflows/     # CI configuration
```

### Key Functions and APIs
- `psis(log_likelihood)` - Pareto smoothed importance sampling
- `psis_loo(log_likelihood)` - PSIS leave-one-out cross-validation  
- `loo_compare(models...)` - Compare multiple models
- `pointwise_log_likelihoods(model, chain)` - Extract log-likelihoods from Turing models

### Development Workflow
1. Make changes to source files in `src/`
2. Test changes: `julia --project=. -e "using Pkg; Pkg.test()"` (2 minutes)
3. Update documentation if needed
4. Validate with manual test scenarios
5. Ensure all tests pass before committing

### Integration Points
- **Turing.jl Integration**: Extensions handle model introspection and log-likelihood extraction
- **MCMCChains.jl**: Support for MCMC chain analysis and diagnostics
- **R Reference Data**: Tests validate against R's `loo` package implementations

### Performance Expectations
- Package loading: ~2 seconds
- Basic PSIS calculation: <1 second for moderate datasets
- Full LOO-CV: Scales with dataset size and MCMC samples
- Large datasets (1000+ points, 4000+ samples): May take several minutes

### Common Issues and Solutions
- **"Package not found"** during doc build: Run `julia --project=docs -e "using Pkg; Pkg.develop(PackageSpec(path=pwd()))"`
- **Test timeouts**: NEVER cancel - tests take exactly 6 minutes
- **High Pareto k warnings**: Expected behavior for certain datasets, not an error
- **Memory issues**: PSIS can be memory-intensive for very large datasets

### File Locations for Common Tasks
- Core PSIS algorithm: `src/ImportanceSampling.jl`
- LOO-CV implementation: `src/LeaveOneOut.jl` 
- Model comparison: `src/ModelComparison.jl`
- Test reference data: `test/data/*.RData`
- Turing examples: `docs/src/turing.md`
- CI configuration: `.github/workflows/CI.yml`

**Remember: This is a specialized statistical package. Always validate changes against reference implementations and ensure mathematical correctness.**