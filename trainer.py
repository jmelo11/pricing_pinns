from torch.optim import Optimizer, LBFGS
from torch_kfac import KFAC
from tqdm import tqdm
import torch
from kfac.preconditioner import KFACPreconditioner
from kfac.scheduler import LambdaParamScheduler

from solution import *
from datasets import *
from closures import *
from optimizer import *


class PINNTrainer:
    '''
        Trainer for the PINN.
    '''

    def __init__(self):
        self.optimizer = None
        self.closure = None
        self.epochs = None
        self.scheduler = None
        self.preconditioner = None
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("mps")
        self.dtype = torch.float32

    def with_optimizer(self, optimizer: Optimizer):
        self.optimizer = optimizer
        return self

    def with_dataset(self, dataset: SampledDataset, loader_opts: dict):
        self.dataset = dataset
        self.loader_opts = loader_opts
        return self

    def with_device(self, device: torch.device):
        self.device = device
        return self

    def with_dtype(self, dtype: torch.dtype):
        self.dtype = dtype
        return self

    def with_epochs(self, epochs: int):
        self.epochs = epochs
        return self

    def with_scheduler(self, scheduler):
        self.scheduler = scheduler
        self.scheduler
        return self

    def with_training_step(self, closure: Closure):
        closure.with_device(self.device)\
            .with_dtype(self.dtype)\
            .with_optimizer(self.optimizer)
        self.closure = closure
        return self

    def with_preconditioner(self, preconditioner):
        self.preconditioner = preconditioner
        return self

    def train(self):
        if isinstance(self.optimizer, LBFGS):
            pbar = tqdm(range(self.optimizer.state_dict()
                        ['param_groups'][0]['max_eval']), desc="LBFGS training")

            last_evals = 0

            def closure():
                nonlocal last_evals
                self.optimizer.zero_grad()
                loss = self.closure()
                loss.backward()
                if self.preconditioner:
                    self.preconditioner.step()
                results = compare_with_mc(self.closure.model,
                                          self.closure.dataset.params, n_prices=5, n_simulations=10_000, dtype=self.dtype, device=self.device, seed=44)

                pbar.update(self.optimizer.state_dict()[
                    'state'][0]['func_evals']
                    - last_evals)

                last_evals = self.optimizer.state_dict()[
                    'state'][0]['func_evals']

                pbar.set_postfix({
                    "Interior": f"{self.closure.get_state()['interior_loss'][-1]:.6f}",
                    "Boundary": f"{self.closure.get_state()['boundary_loss'][-1]:.6f}",
                    "Initial": f"{self.closure.get_state()['initial_loss'][-1]:.6f}",
                    "Total": f"{(self.closure.get_state()['interior_loss'][-1]+ self.closure.get_state()['boundary_loss'][-1]+ self.closure.get_state()['initial_loss'][-1]): .6f}",
                    "Max Error": f"{results['max_error']:.6f}",
                    "L2 Error": f"{results['l2_rel_error']:.6f}",

                })
                return loss

            self.closure.next_batch()
            self.optimizer.step(closure)
            pbar.close()

        elif isinstance(self.optimizer, Adahessian):
            pbar = tqdm(range(self.epochs), desc="Adahessian training")
            for epoch in pbar:
                self.closure.next_batch()
                self.optimizer.zero_grad()
                loss = self.closure()
                loss.backward(create_graph=True)
                self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()

                results = compare_with_mc(self.closure.model,
                                          self.closure.dataset.params, n_prices=5, n_simulations=10_000, dtype=self.dtype, device=self.device, seed=44)

                pbar.set_postfix({
                    "Interior": f"{self.closure.get_state()['interior_loss'][-1]:.6f}",
                    "Boundary": f"{self.closure.get_state()['boundary_loss'][-1]:.6f}",
                    "Initial": f"{self.closure.get_state()['initial_loss'][-1]:.6f}",
                    "Total": f"{(self.closure.get_state()['interior_loss'][-1]+ self.closure.get_state()['boundary_loss'][-1]+ self.closure.get_state()['initial_loss'][-1]): .6f}",
                    "Max Error": f"{results['max_error']:.6f}",
                    "L2 Error": f"{results['l2_rel_error']:.6f}",
                })
            pbar.close()

        # elif isinstance(self.optimizer, KFAC):
        #     pbar = tqdm(range(self.epochs), desc="KFAC training")
        #     for epoch in pbar:
        #         self.closure.next_batch()
        #         self.closure.model.zero_grad()
        #         with self.optimizer.track_forward():
        #             loss = self.closure()
        #         with self.optimizer.track_backward():
        #             loss.backward()
        #         self.optimizer.step(loss=loss)

        #         results = compare_with_mc(self.closure.model,
        #                                   self.closure.dataset.params, n_prices=5, n_simulations=10_000, dtype=self.dtype, device=self.device, seed=44)

        #         pbar.set_postfix({
        #             "Interior": f"{self.closure.get_state()['interior_loss'][-1]:.6f}",
        #             "Boundary": f"{self.closure.get_state()['boundary_loss'][-1]:.6f}",
        #             "Initial": f"{self.closure.get_state()['initial_loss'][-1]:.6f}",
        #             "Total": f"{(self.closure.get_state()['interior_loss'][-1]+ self.closure.get_state()['boundary_loss'][-1]+ self.closure.get_state()['initial_loss'][-1]): .6f}",
        #             "Max Error": f"{results['max_error']:.6f}",
        #             "L2 Error": f"{results['l2_rel_error']:.6f}",
        #         })

        elif isinstance(self.optimizer, NysNewtonCG):
            pbar = tqdm(range(self.epochs), desc="NysNewtonCG training")
            precond_update_freq = 20

            def closure(save_progress=True):
                self.optimizer.zero_grad()
                loss = self.closure(save_progress=save_progress)
                grad_tuple = torch.autograd.grad(
                    loss, self.closure.model.parameters(), create_graph=True)
                return loss, grad_tuple

            for epoch in pbar:
                self.closure.next_batch()
                # Update the preconditioner for NysNewtonCG
                if epoch % precond_update_freq == 0:
                    loss, grad_tuple = closure(False)
                    self.optimizer.update_preconditioner(grad_tuple)

                _ = self.optimizer.step(closure)

                results = compare_with_mc(self.closure.model,
                                          self.closure.dataset.params, n_prices=5, n_simulations=10_000, dtype=self.dtype, device=self.device, seed=44)

                pbar.set_postfix({
                    "Interior": f"{self.closure.get_state()['interior_loss'][-1]:.6f}",
                    "Boundary": f"{self.closure.get_state()['boundary_loss'][-1]:.6f}",
                    "Initial": f"{self.closure.get_state()['initial_loss'][-1]:.6f}",
                    "Total": f"{(self.closure.get_state()['interior_loss'][-1]+ self.closure.get_state()['boundary_loss'][-1]+ self.closure.get_state()['initial_loss'][-1]): .6f}",
                    "Max Error": f"{results['max_error']:.6f}",
                    "L2 Error": f"{results['l2_rel_error']:.6f}",

                })
            pbar.close()
        else:
            pbar = tqdm(range(self.epochs), desc="Adam training")

            for epoch in pbar:
                self.closure.next_batch()
                self.optimizer.zero_grad()
                loss = self.closure()
                loss.backward()
                if self.preconditioner:
                    self.preconditioner.step()
                self.optimizer.step()

                results = compare_with_mc(self.closure.model,
                                          self.closure.dataset.params, n_prices=5, n_simulations=10_000, dtype=self.dtype, device=self.device, seed=44)

                if self.scheduler:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(results['max_error'])
                    else:
                        self.scheduler.step()

                pbar.set_postfix({
                    "Interior": f"{self.closure.get_state()['interior_loss'][-1]:.6f}",
                    "Boundary": f"{self.closure.get_state()['boundary_loss'][-1]:.6f}",
                    "Initial": f"{self.closure.get_state()['initial_loss'][-1]:.6f}",
                    "Total": f"{(self.closure.get_state()['interior_loss'][-1]+ self.closure.get_state()['boundary_loss'][-1]+ self.closure.get_state()['initial_loss'][-1]): .6f}",
                    "Max Error": f"{results['max_error']:.10f}",
                    "L2 Error": f"{results['l2_rel_error']:.10f}",
                })
            pbar.close()


if __name__ == "__main__":
    '''
        Usage example.
    '''
    from nn import *
    from utils import *
    import plotly.graph_objects as go

    # Global parameters
    assets = 10
    interior_samples = 10_000
    initial_samples = 10_000
    boundary_samples = 10_000

    sampler = "Hammersley"  # ["LHS", "Halton", "Hammersley", "Sobol"]:
    device = torch.device("cpu")
    dtype = torch.float32
    nn_shape = "32x3"
    # Define option valuation params
    params = OptionParameters(
        n_assets=assets,
        tau=1.0,
        sigma=np.array([0.2] * assets),
        rho=np.eye(assets) + 0.25 * (np.ones((assets, assets)
                                             ) - np.eye(assets)),
        r=0.05,
        strike=60,
        payoff=payoff
    )

    # Create dataset to traing over
    dataset = SampledDataset(
        params, interior_samples, initial_samples, boundary_samples, sampler, dtype, device)

    # Build NN
    # model = build_nn(
    #     nn_shape=nn_shape,
    #     input_dim=assets,
    #     dtype=torch.float32
    # )

    model = NNAnzats(n_layers=6, input_dim=assets+1,
                     hidden_dim=32, output_dim=1)
    model.to(device)
    model.train()

    # Set optimizer and training function

    # optimizer = LBFGS(
    #     model.parameters(),
    #     max_iter=100,
    #     max_eval=5_000,
    #     line_search_fn="strong_wolfe",
    # )
    # batch_size = len(dataset)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3)
    batch_size = 100
    total_iter = 10_000
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, total_iters=total_iter)

    closure = DimlessBS()\
        .with_dataset(dataset, loader_opts={'batch_size': batch_size, "shuffle": True, "pin_memory": True})\
        .with_model(model)\
        .with_device(device)\
        .with_dtype(dtype)

    # closure = WeightBalancingNonDimBlackScholesClosure()\
    #     .with_dataset(dataset, loader_opts={'batch_size': batch_size, "shuffle": True, "pin_memory": True})\
    #     .with_model(model)\
    #     .with_device(device)\
    #     .with_dtype(dtype)

    trainer = PINNTrainer()\
        .with_optimizer(optimizer)\
        .with_device(device)\
        .with_dtype(dtype)\
        .with_training_step(closure)\
        .with_epochs(total_iter)\
        .with_scheduler(scheduler)

    trainer.train()

    state = trainer.closure.get_state()
    plot_loss(state, nn_shape)
