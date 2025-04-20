from torch.optim import Optimizer
from tqdm import tqdm
import torch
import os

from derpinns.solution import *
from derpinns.datasets import *
from derpinns.closures import *
from derpinns.optimizer import *


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
        if isinstance(self.optimizer, LBFGS) or isinstance(self.optimizer, BFGS) or isinstance(self.optimizer, SSBroyden):
            pbar = tqdm(range(self.optimizer.state_dict()
                              ['param_groups'][0]['max_iter']), desc=str(self.optimizer.__class__.__name__) + " training")

            last_evals = 0

            def closure(update_status=True):
                nonlocal last_evals
                self.optimizer.zero_grad()
                loss = self.closure(update_status=update_status)
                loss.backward()

                if update_status:
                    results = compare_with_mc(self.closure.model,
                                              self.closure.dataset.params, n_prices=5, n_simulations=10_000, dtype=self.dtype, device=self.device, seed=44)
                    pbar.update(self.optimizer.state_dict()[
                        'state'][0]['n_iter']
                        - last_evals)

                    last_evals = self.optimizer.state_dict()[
                        'state'][0]['n_iter']

                    pbar.set_postfix({
                        "Interior": f"{self.closure.get_state()['interior_loss'][-1]:.6f}",
                        "Boundary": f"{self.closure.get_state()['boundary_loss'][-1]:.6f}",
                        "Initial": f"{self.closure.get_state()['initial_loss'][-1]:.6f}",
                        "Total": f"{(self.closure.get_state()['interior_loss'][-1]+ self.closure.get_state()['boundary_loss'][-1]+ self.closure.get_state()['initial_loss'][-1]): .6f}",
                        "Max Error": f"{results['max_error']:.6f}",
                        "L2 Error": f"{results['l2_rel_error']:.6f}",
                    })
                    self.closure.update_errors_state(
                        results['max_error'], results['l2_rel_error'])
                    # self.closure.log_state()
                return loss

            self.closure.next_batch()
            self.optimizer.step(closure)
            pbar.close()

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
                self.closure.update_errors_state(
                    results['max_error'], results['l2_rel_error'])
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

                self.closure.update_errors_state(
                    results['max_error'], results['l2_rel_error'])

            pbar.close()
