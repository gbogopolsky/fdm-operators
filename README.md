# fdm-operators
PyTorch and NumPy implementation of the physical operators (gradient, divergence, curl, laplacian) in Finite Difference formulation over rectangular cartesian grids.

## Tests
Tests are implemented in `pytest`. To run them, call `pytest` from the root directory.  
Each test script can also generate the verification plots, e.g. by calling `python tests/test_divergence.py` from the root directory. 
