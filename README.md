# Matter Perturbations with a Physics-Informed Neural Network (PINN)

A neural network trained without labeled data to solve a differential equation across a range of parameters, rather than running a numerical solver for each parameter configuration.

![PINN compared with the numerical solution](figures/comparison.png)

## The problem: the evolution of large-scale structure

In cosmology, the growth of matter perturbations, which underlies the formation of galaxies and large-scale structure, is described by a second-order differential equation. Testing a cosmological model against observations requires solving this equation repeatedly for different parameter combinations. This can be computationally expensive when fitting parameters or analyzing stability.

A numerical integrator, such as a Runge–Kutta solver, provides a reference solution, but the integration must be repeated whenever the parameters change.

## The idea: train once

Instead of solving the equation separately for each parameter value, a neural network learns **an entire family of solutions** at once: not just `δ(a)` for a fixed `Ω_m0`, but `δ(a, Ω_m0)` over a range of `Ω_m0` values simultaneously. This is known as a *bundle solution*.

No labeled training data are used. The loss function is the squared residual of the differential equation, averaged over points sampled from the domain. Training therefore encourages the network to satisfy the governing equation.

Once trained, evaluating the model for a new `Ω_m0` within the training range requires only a forward pass through the network.

## Results

The repository includes two trained networks, with their weights saved in `models/`:

- **`single_om030`**: a network trained for a fixed `Ω_m0 = 0.3`.
- **`bundle_om_010_050`**: a bundle network trained over `Ω_m0 ∈ [0.1, 0.5]`.

Both are validated against an RK45 reference integrator implemented with SciPy's `solve_ivp` in `cosmopinn/reference.py`.

![Relative error versus a for Ω_m0 = 0.3](figures/error_curve.png)

![Error across the full Ω_m0 range](figures/error_heatmap.png)

## How to run

Run the following commands from the repository root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest -q                                          # Test the saved model weights

python scripts/train_single.py                     # Train at fixed Ω_m0 (~1–2 min on CPU)
python scripts/train_bundle.py                     # Train the bundle (~15 min on CPU)
python scripts/validate.py --run models/bundle_om_010_050
```

The notebook `notebooks/perturbaciones_pinn.ipynb` walks through configuration, training, and validation. It can use the saved weights to validate the results without retraining, running in less than a minute.

## Validation

- Predictions are compared with a numerical integrator (`scipy.solve_ivp`, RK45) that solves the same equation without a change of variables.
- `cosmopinn/cosmology.py` defines the background cosmology. Both the neural network and the reference solver use the same functions, keeping the physical model consistent across implementations.
- Automated tests (`pytest`, run in CI on each push) check the physical limits of the background cosmology and the reference integrator's behavior. A regression test loads the saved model weights and checks prediction errors against the numerical reference.

## Repository structure

```text
pinns_for_cosmology/
├── cosmopinn/
│   ├── cosmology.py     # ΛCDM background: E(a)², d(ln H)/dN
│   ├── equation.py      # Change of variables and ODE residual
│   ├── training.py      # Build the network and neurodiffeq solver; train and save
│   ├── reference.py     # Reference solution (scipy.solve_ivp)
│   └── evaluate.py      # Load a trained run and compute error metrics
├── scripts/
│   ├── train_single.py  # Fixed Ω_m0
│   ├── train_bundle.py  # Bundle over Ω_m0
│   └── validate.py      # Compare the PINN with the reference solution
├── models/             # Saved weights and outputs (nets.pth, config.json, loss.npy)
├── figures/            # Figures used in this README
├── notebooks/          # Annotated notebook covering the full workflow
└── tests/
```

## Limitations and further work

This repository demonstrates the method in its simplest form: the standard cosmological model (ΛCDM) and a single bundle parameter (`Ω_m0`). It represents part of my thesis work, which extends the same approach to modified gravity (`f(R)`, the Hu–Sawicki model) and a four-parameter bundle. Those extensions belong to the broader thesis project.

## Context

This project presents a self-contained example from my master's thesis in Physics on Physics-Informed Neural Networks applied to cosmology. The code was rewritten from scratch for clarity and readability. It implements the same physics as the thesis work, rather than copying the original thesis repository.

Built with [PyTorch](https://pytorch.org/) and [neurodiffeq](https://github.com/NeuroDiffGym/neurodiffeq).
