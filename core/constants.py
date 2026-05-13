from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config512D:
    """All hyperparameters for the 512D modular field system.

    Call validate() after construction to assert structural invariants.
    """

    seed: int = 42

    # Dimensions
    dim: int = 540
    dim_f: int = 480           # field space (num_modules * module_size)
    dim_c: int = 60            # one constraint per module activation

    # Module topology
    num_modules: int = 60
    module_size: int = 8
    input_dim: int = 64
    output_dim: int = 64

    # Dynamics
    alpha: float = 0.015       # memory weight (X_{t-1} term)
    epsilon: float = 0.07      # kernel mixing rate
    lambda_f: float = 0.84     # field persistence (diagonal decay)
    lambda_c: float = 0.45     # constraint persistence

    # Coupling
    intra_coupling: float = 0.018    # within-module circulant coupling
    inter_coupling: float = 0.012    # between-module exponential coupling
    inter_mix_noise: float = 0.25    # mixed-dim inter-module block noise
    inter_rank: int = 16             # rank of K_inter low-rank factorization (U @ V.T)

    # Spectral guarantee
    target_radius: float = 0.94

    # Input injection
    input_strength: float = 0.18

    # Routing
    route_temperature: float = 0.85
    active_route_floor: float = 0.004
    k_winners: int = 5               # number of modules active per input (0 = full softmax)

    # Learnable K_inter
    inter_lr: float = 0.0005         # Hebbian learning rate for U, V factors
    inter_renorm_every: int = 50     # renormalize spectral scale every N training samples
    learnable_inter_modules: bool = True  # update K_inter factors during single-node training

    # Governance / projection
    enable_projection: bool = True
    boundary_theta: float = 4.0      # energy norm threshold
    boundary_delta: float = 0.75     # norm rate-of-change threshold
    projection_damping: float = 0.55
    learnable_constraint_feedback: bool = True
    constraint_feedback_lr: float = 0.0008
    constraint_feedback_clip: float = 0.5

    # Training
    readout_lr: float = 0.035
    router_lr: float = 0.003
    specialization_lr: float = 0.002
    train_samples: int = 120
    train_epochs: int = 45
    process_steps: int = 18

    # Analysis
    propagation_threshold: float = 0.012

    def validate(self) -> None:
        assert self.dim_f == self.num_modules * self.module_size, (
            f"dim_f ({self.dim_f}) must equal num_modules * module_size "
            f"({self.num_modules} * {self.module_size})"
        )
        assert self.dim == self.dim_f + self.dim_c, (
            f"dim ({self.dim}) must equal dim_f + dim_c ({self.dim_f} + {self.dim_c})"
        )
        assert self.input_dim >= self.module_size
        assert self.output_dim <= self.dim
        assert self.dim_c == self.num_modules, (
            "dim_c must equal num_modules because constraints are module activations"
        )
