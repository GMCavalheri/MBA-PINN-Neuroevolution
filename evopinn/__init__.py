"""EvoPINN: single-generation neuroevolution of PINN architectures.

Code for the article "Applying Neuroevolution to the Architecture Search of
Physics-Informed Neural Networks" (Cavalheri & Contreras).
"""
import os

# Must be set before the first cuBLAS call for deterministic GPU results.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

__version__ = "1.0.0"
