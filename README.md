# Beam Search Decoder for Quantum LDPC Codes

This repository contains the source code required to reproduce the simulation results presented in the paper:
**[Beam Search Decoder for Quantum LDPC Codes](https://arxiv.org/abs/2512.07057)**.

## Reproduction Instructions

To reproduce the results, follow these 3 steps:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Build the Decoder Extension:**
   Navigate to the `decoder` directory and compile the C++ extension:
   ```bash
   cd decoder
   python3 setup.py build_ext --inplace
   ```

3. **Run the Simulations:** Open `Beam_Search.ipynb` and execute the cells following the instructions provided within the notebook.

## File Overview

* `StimCircuit/`: Stim circuits used for simulation.

* `simulation_results/`: Directory for storing simulation outputs.

* `decoder/src_cpp/`: C++ source files (main file: `beam_search.hpp`).

* `decoder/beam_search_decoder/`: Python wrappers.

* `Beam_Search.ipynb`: Notebook to reproduce paper results.

* `simulation_functions.py`: Python scripts that are used in `Beam_Search.ipynb`

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{ye2026beam,
  title={Beam Search Decoder for Quantum Low-Density Parity-Check Codes},
  author={Ye, Min and Wecker, Dave and Delfosse, Nicolas},
  journal={PRX Quantum},
  volume={7},
  number={3},
  pages={033002},
  year={2026},
  publisher={APS}
}
```

## License
Copyright (c) 2025 IonQ, Inc., all rights reserved

This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## Third-Party Components

This project includes code from the ldpc library (https://github.com/quantumgizmos/ldpc), originally released under the MIT License.

The files `/decoder/src_cpp/gf2sparse.hpp` and `/decoder/src_cpp/sparse_matrix_base.hpp` are MIT-licensed and retain that license.
