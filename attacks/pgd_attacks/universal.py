import torch
from attacks.pgd_attacks.attack import Attack
import numpy as np
from matplotlib import pyplot as plt

class UPGD(Attack):
    def __init__(self, model, criterion, misc_args=None, pgd_args=None):
        super(UPGD, self).__init__(model, criterion, misc_args, pgd_args)
        self.alpha = pgd_args['alpha']
        self.eps = pgd_args['eps']
        self.n_iter = pgd_args['n_iter']
        self.n_restarts = pgd_args['n_restarts']
        self.rand_init = pgd_args['rand_init']
        self.device = misc_args['device']

        # Broadcast self.data_RGB_start and self.data_RGB_end to [C, H, W]
        data_shape = misc_args['data_shape']
        self.data_RGB_start = torch.tensor(misc_args['data_RGB_start'], device=self.device).view(-1, 1, 1).expand(data_shape)
        self.data_RGB_end = torch.tensor(misc_args['data_RGB_end'], device=self.device).view(-1, 1, 1).expand(data_shape)

        self.name = 'UPGD'
    def report_schematics(self):

        print("Attack L_inf norm limitation:")
        print(self.eps_ratio)
        print("Number of iterations for perturbation optimization:")
        print(self.n_iter)
        print("Number of restarts for perturbation optimization:")
        print(self.n_restarts)

    def perturb(self, x, y, targeted=False, batch_size=64):
        """
        Calculate a universal perturbation for the dataset using batch processing.
        Args:
            x: Input data tensor, shape (N, C, H, W).
            y: Target labels, shape (N,).
            targeted: If True, perform a targeted attack; otherwise, untargeted.
            batch_size: Number of samples to process in each batch.
        Returns:
            universal_pert: Universal perturbation tensor, shape (C, H, W).
            adv_pert_loss: Final loss after applying the universal perturbation.
        """
        mean_loss_per_step = []

        # Initialize universal perturbation with the same dimensions as inputs
        universal_pert = torch.zeros_like(x[0], device=self.device)
        best_mean_loss = 0.0 # Start from perfect loss - we want to maximize it

        self.model.eval()

        for rest in range(self.n_restarts):
            # Initialize perturbation for this restart
            if self.rand_init:
                # print("RANDOM INITIALIZATION " * 10)
                # pert = self.generate_fourier_noise(universal_pert.shape, self.eps)
                pert = torch.empty_like(universal_pert).uniform_(-self.eps, self.eps)
                # pert = torch.empty_like(universal_pert).normal_(mean=0, std=self.eps / 4).clamp(-self.eps, self.eps)
            else:
                # print("ZERO INITIALIZATION " * 10)
                pert = universal_pert.clone()

            pert.requires_grad = True  # Ensure gradient tracking

            for _ in range(self.n_iter):
                total_loss = 0.0

                for batch_start in range(0, len(x), batch_size):
                    # Process a batch of samples
                    batch_end = min(batch_start + batch_size, len(x))
                    batch_x = x[batch_start:batch_end].to(self.device)  # Batch of inputs
                    batch_y = y[batch_start:batch_end].to(self.device)  # Corresponding labels

                    # Apply the universal perturbation
                    perturbed_x = torch.clamp(batch_x + pert, self.data_RGB_start, self.data_RGB_end)

                    # Compute loss for the batch
                    output = self.model(perturbed_x)
                    loss = self.criterion(output, batch_y).mean()
                    total_loss += loss.item()

                    # Compute gradient
                    grad = torch.autograd.grad(loss, pert, retain_graph=False)[0]
                    grad = grad.mean(dim=0)  # Aggregate gradients across the batch

                    # Update perturbation
                    with torch.no_grad():
                        pert += self.alpha * grad.sign()
                        pert = torch.clamp(pert, -self.eps, self.eps)  # Enforce L_inf constraint
                        pert = torch.clamp(batch_x + pert, self.data_RGB_start, self.data_RGB_end).mean(dim=0) - batch_x.mean(dim=0)
                    # Re-enable requires_grad for the next iteration
                    pert.requires_grad = True
                mean_loss = total_loss / len(x)
                mean_loss_per_step.append(mean_loss)
                # Update the universal perturbation if it improves performance
                if mean_loss > best_mean_loss:
                    best_mean_loss = mean_loss
                    universal_pert = pert.clone().detach()  # Detach to save the best perturbation
                    print(f"Restart {rest + 1}/{self.n_restarts}, Iteration {_ + 1}/{self.n_iter}, Mean Loss: {best_mean_loss}")
        # clamp perturbation to epsilon
        # universal_pert = torch.clamp(universal_pert, -self.eps, self.eps)
        universal_pert.clamp_(-self.eps, self.eps)
        print(f"EPSILON: {self.eps}")
        print(f"MAX PERT: {universal_pert.abs().max()}")
        print(f"TEST: {universal_pert.abs().max() <= self.eps}")
        return universal_pert, best_mean_loss, mean_loss_per_step