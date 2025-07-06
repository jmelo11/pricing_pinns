from typing import List, Dict, Tuple
from enum import Enum
from typing import List
import torch.nn as nn
from typing import List, Tuple
import torch.nn.functional as F
from torch.autograd import grad
from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from derpinns.collocations import *
from derpinns.datasets import *
from derpinns.sampling import residual_based_adaptive_sampling
from torch.func import jacrev, jacfwd, vmap

torch.autograd.set_detect_anomaly(True)


class Closure(ABC):
    """
          Base class for the closure (single training step) to be used by the optimizer.
    """

    def __init__(self):
        self.dataset = None
        self.model = None

        self.x = None
        self.y = None
        self.mask = None

        self.dtype = None
        self.device = None
        self.optimizer = None

        self.state = {
            "interior_loss": [],
            "boundary_loss": [],
            "initial_loss": [],
            "max_err": [],
            "l2_rel_err": [],
        }

    def with_device(self, device: torch.device):
        if device:
            self.device = device
            return self
        else:
            raise ValueError("Invalid device")

    def get_state(self):
        return self.state

    def with_dtype(self, dtype: torch.dtype):
        if dtype:
            self.dtype = dtype
            return self
        else:
            raise ValueError("Invalid dtype")

    def with_optimizer(self, optimizer: Optimizer):
        if optimizer:
            self.optimizer = optimizer
            return self
        else:
            raise ValueError("Invalid optimizer")

    def next_batch(self):
        self.x, self.y, self.mask = next(iter(self.dataloader))

    def update_losses_state(self, pde_loss, boundary_loss, initial_cond_loss):
        self.state["interior_loss"].append(pde_loss)
        self.state["boundary_loss"].append(boundary_loss)
        self.state["initial_loss"].append(initial_cond_loss)

    def update_errors_state(self, max_err, l2_err):
        self.state["max_err"].append(max_err)
        self.state["l2_rel_err"].append(l2_err)

    def log_state(self):
        print(f"Interior Loss: {self.state['interior_loss'][-1]}")
        print(f"Boundary Loss: {self.state['boundary_loss'][-1]}")
        print(f"Inital Condition Loss: {self.state['initial_loss'][-1]}")
        print(
            f"Total Loss: {self.state['interior_loss'][-1]+self.state['boundary_loss'][-1]+self.state['initial_loss'][-1]}")
        print("-"*40)

    @abstractmethod
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        pass

    def with_dataset(self, dataset: SampledDataset, loader_opts: dict):
        self.dataset = dataset
        self.dataloader = DataLoader(
            dataset,
            **loader_opts
        )
        return self

    def with_model(self, model: torch.nn.Module):
        if model:
            self.model = model
            return self
        else:
            raise ValueError("Invalid Model")


class DimlessBS(Closure):
    """
        Training step of the non-dimensional Black-Scholes PDE.
    """

    def __init__(self):
        super().__init__()
        self.n_assets = None

    def with_dataset(self, dataset: SampledDataset, loader_opts: dict):
        super().with_dataset(dataset, loader_opts)
        self.n_assets = dataset.params.n_assets
        self.sigma = dataset.params.sigma
        self.r = dataset.params.r
        self.rho = dataset.params.rho
        return self

    def compute_derivatives(self, x) -> tuple:
        """
            Computes all required derivatives using autograd.
        """
        x.requires_grad_(True)
        u = self.model(x)
        grads = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_tau = grads[:, -1]
        u_x = grads[:, :self.n_assets]
        u_xx_list = []
        for j in range(self.n_assets):
            grad_j = torch.autograd.grad(u_x[:, j].sum(), x, create_graph=True,
                                         retain_graph=True)[0][:, :self.n_assets]
            u_xx_list.append(grad_j)
        u_xx = torch.stack(u_xx_list, dim=1)

        if torch.isnan(u_tau).any():
            raise ValueError("NaN @ u_tau")
        if torch.isnan(u_x).any():
            raise ValueError("NaN @ u_x")
        if torch.isnan(u_xx).any():
            raise ValueError("NaN @ u_xx")
        if torch.isnan(u).any():
            raise ValueError("NaN @ u")

        return u, u_tau, u_x, u_xx

    def interior_residual(self, u, u_tau, u_x, u_xx) -> torch.Tensor:
        diffusion = torch.zeros(
            u.shape[0], dtype=self.dtype, device=self.device)
        drift = torch.zeros(u.shape[0], dtype=self.dtype, device=self.device)
        for i in range(self.n_assets):
            diffusion += 0.5 * self.sigma[i]**2 * u_xx[:, i, i]
            drift += (self.r - 0.5 * self.sigma[i]**2) * u_x[:, i]
            for j in range(i+1, self.n_assets):
                diffusion += self.sigma[i] * self.sigma[j] * \
                    self.rho[i, j] * u_xx[:, i, j]

        reaction = self.r * u.squeeze()
        residual = u_tau - diffusion - drift + reaction
        return residual

    def __top_i_boundary_residual(self, u, u_tau, u_x, u_xx, i) -> torch.Tensor:
        drift_sum = torch.zeros(
            u_x.shape[0], dtype=self.dtype, device=self.device)
        for j in range(self.n_assets):
            drift_sum += self.r * u_x[:, j]

        cross_diffusion = torch.zeros(
            u_xx.shape[0], dtype=self.dtype, device=self.device)
        for k in range(self.n_assets):
            for j in range(k+1, self.n_assets):
                if k != i and j != i:
                    cross_diffusion += 2 * self.sigma[k] * \
                        self.sigma[j] * self.rho[k, j] * u_xx[:, k, j]

        pure_diffusion = torch.zeros(
            u_xx.shape[0],  dtype=self.dtype, device=self.device)
        for j in range(self.n_assets):
            if j != i:
                pure_diffusion += self.sigma[j]**2 * \
                    (u_xx[:, j, j] - u_x[:, j])

        residual = -u_tau + drift_sum + 0.5 * \
            (cross_diffusion + pure_diffusion) - self.r * u.squeeze()

        return residual

    def __bottom_i_boundary_residual(self, u, u_tau, u_x, u_xx, i):
        drift_sum = torch.zeros(
            u_x.shape[0], dtype=self.dtype, device=self.device)
        for j in range(self.n_assets):
            if j != i:
                drift_sum += self.r * u_x[:, j]

        cross_diffusion = torch.zeros(
            u_xx.shape[0], dtype=self.dtype, device=self.device)
        for k in range(self.n_assets):
            for j in range(k+1, self.n_assets):
                if k != i and j != i:
                    cross_diffusion += 2 * self.sigma[k] * \
                        self.sigma[j] * self.rho[k, j] * u_xx[:, k, j]

        pure_diffusion = torch.zeros(
            u_xx.shape[0], dtype=self.dtype, device=self.device)
        for j in range(self.n_assets):
            if j != i:
                pure_diffusion += self.sigma[j]**2 * \
                    (u_xx[:, j, j] - u_x[:, j])

        residual = -u_tau + drift_sum + 0.5 * \
            (cross_diffusion + pure_diffusion) - self.r * u.squeeze()
        return residual

    def boundary_loss(self, u, u_tau, u_x, u_xx) -> list[torch.Tensor]:
        """
            Computes the loss of all boundaries (top and bottom).
        """
        losses = torch.zeros(
            self.n_assets*2, dtype=self.dtype, device=self.device)
        for i in range(self.n_assets):
            mask_top_i = self.mask[:, 2 + 2*i].bool()
            mask_bottom_i = self.mask[:, 3 + 2*i].bool()

            # Bottom boundary
            if mask_bottom_i.sum() > 0:
                br = self.__bottom_i_boundary_residual(
                    u[mask_bottom_i],
                    u_tau[mask_bottom_i],
                    u_x[mask_bottom_i],
                    u_xx[mask_bottom_i],
                    i
                )
                losses[2*i] = br.square().mean()

            # Top boundary
            if mask_top_i.sum() > 0:
                tr = self.__top_i_boundary_residual(
                    u[mask_top_i],
                    u_tau[mask_top_i],
                    u_x[mask_top_i],
                    u_xx[mask_top_i],
                    i
                )
                losses[2*i + 1] = tr.square().mean()
        return losses

    def initial_residual(self, u, y) -> torch.Tensor:
        return u - y

    def compute_losses(self):
        """
            The training step. Computes all losses and returns the total.
        """
        # self.optimizer.zero_grad()
        u, u_tau, u_x, u_xx = self.compute_derivatives(self.x)

        # PDE residual: interi or loss
        interior_mask = self.mask[:, 0].bool()
        if interior_mask.sum() > 0:
            interior_loss = self.interior_residual(
                u[interior_mask],
                u_tau[interior_mask],
                u_x[interior_mask],
                u_xx[interior_mask]
            ).square().mean()
        else:
            interior_loss = torch.tensor(
                0.0, dtype=self.dtype, device=self.device)

        # Initial condition loss
        initial_mask = self.mask[:, 1].bool()
        if initial_mask.sum() > 0:
            initial_loss = self.initial_residual(
                u[initial_mask],
                self.y[initial_mask]
            ).square().mean()
        else:
            initial_loss = torch.tensor(
                0.0, dtype=self.dtype, device=self.device)

        boundary_losses = self.boundary_loss(
            u, u_tau, u_x, u_xx)

        return interior_loss, boundary_losses, initial_loss

    def __call__(self, *args, **kwargs):
        """
            Wrapper method
        """
        interior_loss, boundary_losses, initial_loss = self.compute_losses()
        boundary_loss = torch.sum(boundary_losses)
        if kwargs.get('update_status', True):
            self.update_losses_state(
                interior_loss.item(), boundary_loss.item(), initial_loss.item())
        return interior_loss + boundary_loss + initial_loss


class ResidualBasedAdaptiveSamplingDimlessBS(DimlessBS):
    """
        Implementation of the residual-based adaptive sampling method.
        https://www.sciencedirect.com/science/article/pii/S0045782522006260?via%3Dihub

        The main disadvantage of this kind of methods is that is does not allow mini-batching.
    """

    def __init__(self,  sampler='Halton', k=1, c=1, seed=None):
        super().__init__()

        self.sampler = sampler
        self.seed = seed
        self.k = k
        self.c = c

    def next_batch(self):
        # Get the next batch of data
        super().next_batch()

        interior_mask = self.mask[:, 0].bool()
        n_samples = self.x[interior_mask].shape[0]

        def res_f(x):
            x = torch.tensor(x, dtype=self.dtype,
                             device=self.device)
            u, u_tau, u_x, u_xx = self.compute_derivatives(x)
            return self.interior_residual(u, u_tau, u_x, u_xx).detach().cpu().numpy()

        tmp_x = residual_based_adaptive_sampling(
            res_f,
            n_samples,
            self.dataset.params.domain_ranges(),
            k=self.k,
            c=self.c,
            sampler=self.sampler,
            seed=self.seed
        )
        # we need to modify the tmp_x to be in the same range as the original x
        tmp_x = torch.tensor(tmp_x, dtype=self.dtype,
                             device=self.device)

        self.x[interior_mask] = tmp_x


class LossBalancingDimlessBS(DimlessBS):
    """
        Implements ReLoBRaLo:
        https://arxiv.org/abs/2110.09813
    """

    def __init__(self, alpha: torch.Tensor, tau: torch.Tensor, rho_prob: torch.Tensor):
        super().__init__()
        self.alpha = alpha
        self.tau = tau
        self.rho_prob = rho_prob

        self.loss_0 = None
        self.loss_t_1 = None
        self.lambda_t_1 = None

    def __call__(self, *args, **kwargs):
        m = self.n_assets*2+2
        epsilon = 1e-8
        rho = torch.bernoulli(torch.tensor(self.rho_prob))
        interior_loss, boundary_losses, initial_loss = self.compute_losses()

        current_loss = torch.zeros(
            m, dtype=self.dtype, device=self.device)
        current_loss[0] = interior_loss
        current_loss[1] = initial_loss
        current_loss[2:] = boundary_losses
        with torch.no_grad():
            if self.loss_0 is None:
                self.loss_0 = current_loss

            if self.loss_t_1 is None:
                self.loss_t_1 = current_loss

            if self.lambda_t_1 is None:
                self.lambda_t_1 = torch.ones(
                    m, dtype=self.dtype, device=self.device)

            lambda_bal_0 = m * \
                torch.softmax(current_loss/(self.tau*self.loss_0 + epsilon), 0)

            lambda_bal_t_1 = m * \
                torch.softmax(
                    current_loss/(self.tau*self.loss_t_1 + epsilon), 0)

            lambda_hist = rho*self.lambda_t_1+(1-rho)*lambda_bal_0
            self.lambda_t_1 = self.alpha*lambda_hist + \
                (1-self.alpha)*lambda_bal_t_1

        self.loss_t_1 = current_loss
        losses = self.lambda_t_1 * current_loss
        if kwargs.get('update_status', True):
            self.update_losses_state(
                losses[0].item(), losses[2:].sum().item(), losses[1].item())
        return losses.sum()


class _Mode(Enum):
    MANUAL = "MANUAL"
    LR_ANNEALING = "LR_ANNEALING"
    SOFTADAPT = "SOFTADAPT"
    RELOBRALO = "RELOBRALO"
    GRADNORM = "GRADNORM"


class MultiBalanceDimlessBS(DimlessBS):
    """
    A DimlessBS closure that can run one of five loss-balancing rules.

    Parameters
    ----------
    mode : str
        One of "MANUAL", "LR_ANNEALING", "SOFTADAPT",
        "RELOBRALO", "GRADNORM".
    alpha : float
        Smoothing / exponent parameter used by most rules.
    tau   : float
        Temperature (SoftAdapt / ReLoBRaLo).
    rho_prob : float
        Bernoulli probability for ReLoBRaLo.
    """

    # ------------------------------------------------------------------ #
    # construction                                                       #
    # ------------------------------------------------------------------ #
    def __init__(self,
                 mode: str,
                 alpha: float = 0.5,
                 tau:   float = 1.0,
                 rho_prob: float = 0.5,
                 shared_param_id: int = -2):
        DimlessBS.__init__(self)

        self.mode = _Mode(mode.upper())
        self.alpha = alpha
        self.tau = tau          # also called T in SoftAdapt / ReLoBRaLo
        self.rho_prob = rho_prob
        self.shared_id = shared_param_id

        # --- state -----------------------------------------------------
        self._lambda = None         # vector of λᵢ (learnable for GradNorm)
        self.l_hist = None         # last-step losses  (SoftAdapt/ReLoB)
        self.l0_hist = None         # first-step losses (ReLoB)
        self._eps = 1e-8

    # ------------------------------------------------------------------ #
    # utilities                                                          #
    # ------------------------------------------------------------------ #

    def _init_vectors(self, m: int, device, dtype):
        if self._lambda is None:
            if self.mode is _Mode.GRADNORM:
                # GRADNORM – λ are learnable → keep log-space, softplus later
                self._lambda = nn.Parameter(torch.zeros(m, dtype=dtype,
                                                        device=device))
            else:
                # all other rules – λ are plain buffers
                self._lambda = torch.ones(m, dtype=dtype, device=device)

            self.l_hist = torch.ones(m, dtype=dtype, device=device)
            self.l0_hist = torch.ones(m, dtype=dtype, device=device)

    # ------------------------------------------------------------------ #
    # callable                                                           #
    # ------------------------------------------------------------------ #

    def __call__(self, *args, **kwargs) -> torch.Tensor:  # noqa: C901
        # 1) gather raw losses -------------------------------------------------
        m = self.n_assets * 2 + 2
        f_loss, b_losses, n_loss = self.compute_losses()
        current = torch.zeros(m, dtype=self.dtype, device=self.device)
        current[0] = f_loss
        current[1] = n_loss
        current[2:] = b_losses

        self._init_vectors(m, current.device, current.dtype)

        # 2) dispatch by rule --------------------------------------------------
        if self.mode is _Mode.MANUAL:
            total = self._manual(current)

        elif self.mode is _Mode.LR_ANNEALING:
            total = self._lr_annealing(current)

        elif self.mode is _Mode.SOFTADAPT:
            total = self._softadapt(current)

        elif self.mode is _Mode.RELOBRALO:
            total = self._relobralo(current)

        elif self.mode is _Mode.GRADNORM:
            total = self._gradnorm(current)

        else:  # pragma: no cover
            raise ValueError(f"unsupported mode {self.mode}")

        # 3) update history (used by several rules) ----------------------------
        self.l_hist.copy_(current.detach())
        self.l0_hist = torch.where(self.model.training & (self.l0_hist == 1),
                                   current.detach(),
                                   self.l0_hist)

        # 4) status panel ------------------------------------------------------
        if kwargs.get("update_status", True):
            self.update_losses_state(f_loss.item(),
                                     b_losses.sum().item(),
                                     n_loss.item())
        return total

    # --- MANUAL -------------------------------------------------------- #

    def _manual(self, current: torch.Tensor) -> torch.Tensor:
        loss = (self._lambda * current).sum()
        return loss

    # --- LR-ANNEALING -------------------------------------------------- #
    def _lr_annealing(self, current: torch.Tensor) -> torch.Tensor:
        # pick one shared parameter tensor to measure gradients
        W = list(self.model.parameters())[self.shared_id]

        # grads for interior + each boundary loss separately
        grads_f = torch.autograd.grad(current[0], W,
                                      retain_graph=True,
                                      create_graph=True)[0]
        mean_grad_f = grads_f.abs().mean()

        lambs_hat = []
        for j in range(2, len(current)):         # boundary terms only
            gj = torch.autograd.grad(current[j], W,
                                     retain_graph=True,
                                     create_graph=True)[0]
            mean_gj = gj.abs().mean()
            lambs_hat.append(mean_grad_f / (mean_gj + self._eps))

        # EMA update
        self._lambda[2:] = (self.alpha * self._lambda[2:]
                            + (1 - self.alpha) * torch.stack(lambs_hat))

        total = current[0] + (self._lambda[2:] * current[2:]).sum()
        return total

    # --- SOFTADAPT ----------------------------------------------------- #
    def _softadapt(self, current: torch.Tensor) -> torch.Tensor:
        diff = (current - self.l_hist) * self.tau
        lambs_hat = torch.softmax(diff.detach(), 0) * len(current)

        self._lambda = (self.alpha * self._lambda
                        + (1 - self.alpha) * lambs_hat)

        return (self._lambda * current).sum()

    # --- RELOBRALO ----------------------------------------------------- #
    def _relobralo(self, current: torch.Tensor) -> torch.Tensor:
        lambs_hat = torch.softmax(current / (self.l_hist * self.tau + self._eps),
                                  0).detach() * len(current)
        lambs0_hat = torch.softmax(current / (self.l0_hist * self.tau + self._eps),
                                   0).detach() * len(current)

        rho = torch.bernoulli(torch.tensor(self.rho_prob,
                                           device=current.device))
        self._lambda = (rho * self.alpha * self._lambda
                        + (1 - rho) * self.alpha * lambs0_hat
                        + (1 - self.alpha) * lambs_hat)

        return (self._lambda * current).sum()

    # --- GRADNORM ------------------------------------------------------ #
    def _gradnorm(self, current: torch.Tensor) -> torch.Tensor:
        # softplus keeps λᵢ positive
        lambdas = F.softplus(self._lambda) + self._eps

        # primary weighted loss
        L_W = (lambdas * current).sum()

        # gradient norms wrt one shared param
        W = list(self.model.parameters())[self.shared_id]
        GiW = torch.stack([
            torch.autograd.grad(L_W_i, W,
                                retain_graph=True,
                                create_graph=True)[0].norm()
            for L_W_i in lambdas * current
        ])
        GiW_mean = GiW.mean()

        with torch.no_grad():
            li_tilde = current / self.l_hist
            Ri = li_tilde / li_tilde.mean()

        target = GiW_mean * (Ri ** self.alpha)
        L_w = (GiW - target.detach()).abs().sum()

        return L_W + L_w

    # ------------------------------------------------------------------ #
    # expose current λ for logging ------------------------------------- #
    # ------------------------------------------------------------------ #

    def lambda_weights(self) -> torch.Tensor:
        if self._lambda is None:
            raise RuntimeError("call the closure once before accessing λ")
        return (F.softplus(self._lambda) if self.mode is _Mode.GRADNORM
                else self._lambda).detach()


class PINNBoundaryDimlessBS(DimlessBS):
    """
        Uses the solution of the i-1th asset case for the lower boundary.
    """

    def __init__(self):
        super(DimlessBS).__init__()

    def with_dataset(self, dataset: SampledDatasetWithPINNBoundary, loader_opts: dict):
        super().with_dataset(dataset, loader_opts)
        self.n_assets = dataset.params.n_assets
        self.sigma = dataset.params.sigma
        self.r = dataset.params.r
        self.rho = dataset.params.rho
        return self

    def boundary_loss(self, u, u_tau, u_x, u_xx) -> list[torch.Tensor]:
        """
            Instead of computing the lower boundary using the boundary condition, we use the solution of the i-1th asset case, which is already computed,
            so we only calculated the upper boundary and the mse of the bottom boundary.
        """
        losses = torch.zeros(
            self.n_assets*2, dtype=self.dtype, device=self.device)
        for i in range(self.n_assets):
            mask_top_i = self.mask[:, 2 + 2*i].bool()
            mask_bottom_i = self.mask[:, 3 + 2*i].bool()

            # Bottom boundary
            if mask_bottom_i.sum() > 0:
                br = self.y[mask_bottom_i] - u[mask_bottom_i]
                losses[2*i] = br.square().mean()

            # Top boundary
            if mask_top_i.sum() > 0:
                tr = self.__top_i_boundary_residual(
                    u[mask_top_i],
                    u_tau[mask_top_i],
                    u_x[mask_top_i],
                    u_xx[mask_top_i],
                    i
                )
                losses[2*i + 1] = tr.square().mean()
        return losses


class FOPINNClosure(DimlessBS):

    def __init__(self):
        super().__init__()

    def compute_derivatives(self, x):
        # make x a true leaf
        x = x.requires_grad_(True)                # [B, n_assets+1]
        out = self.model(x)                       # [B, 1 + n_assets]
        u, u_hat = out[:, :1], out[:, 1:]

        # 1) true first‐order gradients (space + time)
        grads = torch.autograd.grad(
            u.sum(),
            x,
            create_graph=True,
            retain_graph=True,           # <<< keep the graph alive
        )[0]                                # [B, n_assets+1]
        u_x_true = grads[:, :self.n_assets]
        u_tau_true = grads[:, self.n_assets]

        # 2) predicted first‐order
        u_x_hat = u_hat[:, :self.n_assets]
        u_tau_hat = u_hat[:, self.n_assets]

        # 3) second spatial derivatives
        u_xx_list = []
        for i in range(self.n_assets):
            yi = u_x_hat[:, i].sum()
            full_grad = torch.autograd.grad(
                yi,
                x,
                create_graph=True,
                retain_graph=True,       # <<< keep the graph alive until final backward
            )[0]                        # [B, n_assets+1]
            u_xx_list.append(full_grad[:, :self.n_assets])

        u_xx = torch.stack(u_xx_list, dim=1)  # [B, n_assets, n_assets]
        return u, u_tau_hat, u_x_hat, u_xx, u_tau_true, u_x_true

    def compute_losses(self):
        # get all six pieces
        u, u_tau_hat, u_x_hat, u_xx, u_tau_true, u_x_true = self.compute_derivatives(
            self.x)

        # 1) compatibility MSE between predicted vs true first‐derivs
        comp_time = (u_tau_hat - u_tau_true).square().mean()
        comp_space = (u_x_hat - u_x_true).square().mean()
        compatibility = comp_time + comp_space

        # 2) interior PDE loss (same as before, but using u_tau_hat & u_x_hat & u_xx)
        interior_mask = self.mask[:, 0].bool()
        if interior_mask.any():
            R = self.interior_residual(
                u[interior_mask],
                u_tau_hat[interior_mask],
                u_x_hat[interior_mask],
                u_xx[interior_mask]
            )
            interior_loss = R.square().mean()
        else:
            interior_loss = torch.tensor(0., device=u.device)

        # 3) initial‐condition loss
        init_mask = self.mask[:, 1].bool()
        if init_mask.any():
            I = self.initial_residual(
                u[init_mask],
                self.y[init_mask]
            ).square().mean()
        else:
            I = torch.tensor(0., device=u.device)

        # 4) boundary losses (unchanged)
        B_losses = self.boundary_loss(
            u, u_tau_hat, u_x_hat, u_xx
        )

        return interior_loss, B_losses, I, compatibility

    def compute_losses(self):
        # get all six pieces
        u, u_tau_hat, u_x_hat, u_xx, u_tau_true, u_x_true = self.compute_derivatives(
            self.x)

        # 1) compatibility MSE between predicted vs true first‐derivs
        comp_time = (u_tau_hat - u_tau_true).square().mean()
        comp_space = (u_x_hat - u_x_true).square().mean()
        compatibility = comp_time + comp_space

        # 2) interior PDE loss (same as before, but using u_tau_hat & u_x_hat & u_xx)
        interior_mask = self.mask[:, 0].bool()
        if interior_mask.any():
            R = self.interior_residual(
                u[interior_mask],
                u_tau_hat[interior_mask],
                u_x_hat[interior_mask],
                u_xx[interior_mask]
            )
            interior_loss = R.square().mean()
        else:
            interior_loss = torch.tensor(0., device=u.device)

        # 3) initial‐condition loss
        init_mask = self.mask[:, 1].bool()
        if init_mask.any():
            I = self.initial_residual(
                u[init_mask],
                self.y[init_mask]
            ).square().mean()
        else:
            I = torch.tensor(0., device=u.device)

        # 4) boundary losses (unchanged)
        B_losses = self.boundary_loss(
            u, u_tau_hat, u_x_hat, u_xx
        )

        return interior_loss, B_losses, I, compatibility

    def __call__(self, *args, **kwargs):
        """
        Return the scalar loss for the optimizer.
        """
        interior, B_losses, I, comp = self.compute_losses()
        if kwargs.get('update_status', True):
            self.update_losses_state(
                interior.item(),
                B_losses.sum().item(),
                I.item(),
            )
        return interior + B_losses.sum() + I + comp


class DimlessBSOnlyInterior(DimlessBS):
    """
        Training step of the non-dimensional Black-Scholes PDE using only the interior loss.
    """

    def __init__(self):
        super().__init__()

    def compute_losses(self):
        """
            The training step. Computes all losses and returns the total.
        """
        # self.optimizer.zero_grad()
        u, u_tau, u_x, u_xx = self.compute_derivatives(self.x)

        # PDE residual: interi or loss
        interior_mask = self.mask[:, 0].bool()
        if interior_mask.sum() > 0:
            interior_loss = self.interior_residual(
                u[interior_mask],
                u_tau[interior_mask],
                u_x[interior_mask],
                u_xx[interior_mask]
            ).square().mean()
        else:
            interior_loss = torch.tensor(
                0.0, dtype=self.dtype, device=self.device)

        # Initial condition loss
        initial_mask = self.mask[:, 1].bool()
        if initial_mask.sum() > 0:
            initial_loss = self.initial_residual(
                u[initial_mask],
                self.y[initial_mask]
            ).square().mean()
        else:
            initial_loss = torch.tensor(
                0.0, dtype=self.dtype, device=self.device)

        # to fill the gap so all still works
        boundary_losses = torch.zeros(
            self.n_assets*2, dtype=self.dtype, device=self.device)
        return interior_loss, boundary_losses, initial_loss


class RBABSOnlyInterior(DimlessBS):
    """
        Mix of residual-based adaptive sampling and only interior loss.
    """

    def __init__(self,  sampler='Halton', k=1, c=1, seed=None):
        super().__init__()

        self.sampler = sampler
        self.seed = seed
        self.k = k
        self.c = c

    def next_batch(self):
        # Get the next batch of data
        super().next_batch()

        interior_mask = self.mask[:, 0].bool()
        n_samples = self.x[interior_mask].shape[0]

        def res_f(x):
            x = torch.tensor(x, dtype=self.dtype,
                             device=self.device)
            u, u_tau, u_x, u_xx = self.compute_derivatives(x)
            return self.interior_residual(u, u_tau, u_x, u_xx).detach().cpu().numpy()

        tmp_x = residual_based_adaptive_sampling(
            res_f,
            n_samples,
            self.dataset.params.domain_ranges(),
            k=self.k,
            c=self.c,
            sampler=self.sampler,
            seed=self.seed
        )
        # we need to modify the tmp_x to be in the same range as the original x
        tmp_x = torch.tensor(tmp_x, dtype=self.dtype,
                             device=self.device)

        self.x[interior_mask] = tmp_x

    def compute_losses(self):
        """
            The training step. Computes all losses and returns the total.
        """
        # self.optimizer.zero_grad()
        u, u_tau, u_x, u_xx = self.compute_derivatives(self.x)

        # PDE residual: interi or loss
        interior_mask = self.mask[:, 0].bool()
        if interior_mask.sum() > 0:
            interior_loss = self.interior_residual(
                u[interior_mask],
                u_tau[interior_mask],
                u_x[interior_mask],
                u_xx[interior_mask]
            ).square().mean()
        else:
            interior_loss = torch.tensor(
                0.0, dtype=self.dtype, device=self.device)

        # Initial condition loss
        initial_mask = self.mask[:, 1].bool()
        if initial_mask.sum() > 0:
            initial_loss = self.initial_residual(
                u[initial_mask],
                self.y[initial_mask]
            ).square().mean()
        else:
            initial_loss = torch.tensor(
                0.0, dtype=self.dtype, device=self.device)

        # to fill the gap so all still works
        boundary_losses = torch.zeros(
            self.n_assets*2, dtype=self.dtype, device=self.device)
        return interior_loss, boundary_losses, initial_loss
