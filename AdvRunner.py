import numpy as np
import torch
from tqdm import trange

# Class for running adversarial attacks and evaluations on a given model
class AdvRunner:
    def __init__(self, model, attack, data_RGB_size, device, dtype, verbose=False, threshold=0.5):
        # Initialize the AdvRunner with model, attack configurations, and device settings
        self.attack = attack
        self.model = model
        self.device = device
        self.dtype = dtype
        self.verbose = verbose
        # Reshape data_RGB_size for compatibility with operations
        self.data_RGB_size = torch.tensor(data_RGB_size).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        # Cache attack properties
        self.attack_restarts = self.attack.n_restarts
        self.attack_iter = self.attack.n_iter
        self.attack_report_info = self.attack.report_info
        self.attack_name = self.attack.name

        self.threshold = threshold

    # Evaluate the model on clean inputs
    # def run_clean_evaluation(self, x_orig, y_orig, n_examples, orig_device, batch_size=128):
    #     robust_flags = torch.zeros(n_examples, dtype=torch.bool, device=orig_device)

    #     # Transfer data to the device and process in batches
    #     x = x_orig.clone().detach().to(self.device)
    #     y = y_orig.clone().detach().to(self.device)

    #     # Get model predictions
    #     output = self.model.forward(x)
    #     correct_batch = y.eq(output.max(dim=1)[1]).detach().to(orig_device)
    #     robust_flags = correct_batch

    #     # Calculate initial accuracy
    #     n_robust_examples = torch.sum(robust_flags).item()
    #     init_accuracy = n_robust_examples / n_examples
    #     if self.verbose:
    #         print('initial accuracy: {:.2%}'.format(init_accuracy))
    #     return robust_flags, n_robust_examples, init_accuracy
    
    def run_clean_evaluation(self, x_orig, y_orig, n_examples, orig_device, batch_size=128):
        robust_flags = torch.zeros(n_examples, dtype=torch.bool, device=orig_device)

        # Iterate through the data in batches
        for start_idx in range(0, n_examples, batch_size):
            end_idx = min(start_idx + batch_size, n_examples)

            # Transfer the current batch to the device
            x_batch = x_orig[start_idx:end_idx].clone().detach().to(self.device)
            y_batch = y_orig[start_idx:end_idx].clone().detach().to(self.device)

            # Get model predictions for the current batch
            output_batch = self.model.forward(x_batch)
            correct_batch = y_batch.eq(output_batch.max(dim=1)[1]).detach().to(orig_device)

            # Update robust_flags for the current batch
            robust_flags[start_idx:end_idx] = correct_batch

        # Calculate initial accuracy
        n_robust_examples = torch.sum(robust_flags).item()
        init_accuracy = n_robust_examples / n_examples

        if self.verbose:
            print('initial accuracy: {:.2%}'.format(init_accuracy))

        return robust_flags, n_robust_examples, init_accuracy
    # Process evaluation results
    def process_results(self, n_examples, robust_flags, adv_perts, adv_perts_loss):
        # Calculate robust accuracy, adversarial loss, and maximum perturbation norm
        robust_accuracy = (robust_flags.sum(dim=0) / n_examples).item()
        adv_loss = adv_perts_loss.mean(dim=0).item()
        perts_max_l_inf = (adv_perts.abs() / self.data_RGB_size).view(-1).max(dim=0)[0].item()
        return robust_accuracy, adv_loss, perts_max_l_inf

    # Run universal evaluation with adversarial perturbations
    def run_standard_evaluation(self, x_orig, y_orig, n_examples, batch_size=128):
        with torch.no_grad():
            orig_device = x_orig.device

            # Evaluate on clean examples
            robust_flags, n_robust_examples, init_accuracy = self.run_clean_evaluation(
                x_orig, y_orig, n_examples, orig_device, batch_size
            )

        if self.attack_report_info:
            info_shape = [self.attack_restarts, self.attack_iter + 1, n_examples]
            all_succ = torch.zeros(info_shape, dtype=self.dtype, device=orig_device)
            all_loss = torch.zeros(info_shape, dtype=self.dtype, device=orig_device)
        else:
            all_succ = None
            all_loss = None

        # Generate universal adversarial perturbation
        with torch.cuda.device(self.device):
            x = x_orig.clone().detach()
            y = y_orig.clone().detach()
            # Generate the universal perturbation
            if self.attack_name == 'FourierUPGD':
                universal_pert, adv_loss, mean_loss_per_step = self.attack.perturb(x, y, batch_size=batch_size, threshold=self.threshold)
            else:
                universal_pert, adv_loss, mean_loss_per_step = self.attack.perturb(x, y, batch_size=batch_size)
            # check universal_pert is appropriate norm
            # assert universal_pert.abs().max().item() <= self.attack.eps

        with torch.no_grad():
            perts_max_l_inf = universal_pert.abs().max().item()
            x_adv = torch.zeros_like(x_orig, device=orig_device)
            y_adv = torch.zeros(n_examples, dtype=torch.long, device=orig_device)

            for start_idx in range(0, n_examples, batch_size):
                end_idx = min(start_idx + batch_size, n_examples)
                batch_indices = torch.arange(start_idx, end_idx, device=orig_device)
                # Process the current batch
                x_batch = x_orig[start_idx:end_idx].clone().detach().to(self.device)
                y_batch = y_orig[start_idx:end_idx].clone().detach().to(self.device)

                # Apply universal perturbation to the current batch
                x_adv_batch = x_batch + universal_pert # TODO: check if okay
                # Clamp the adversarial examples to the valid RGB range
                # x_adv_batch = torch.clamp(x_adv_batch, 0, 1) # TODO: this was in the instructions, but check if this is what needed!!!

                # Get predictions on adversarial examples for the current batch
                output_batch = self.model.forward(x_adv_batch)
                y_adv_batch = output_batch.max(dim=1)[1].to(orig_device)
                y_batch = y_batch.to(orig_device)
                x_adv[start_idx:end_idx] = x_adv_batch.to(orig_device)
                y_adv[start_idx:end_idx] = y_adv_batch
                # Update robust flags for incorrect predictions
                false_batch = ~y_batch.eq(y_adv_batch).detach()
                non_robust_indices = batch_indices[false_batch]
                robust_flags[non_robust_indices] = False
                if self.attack_report_info:
                    all_succ[:, :, start_idx:end_idx] = all_batch_succ.to(orig_device)
                    all_loss[:, :, start_idx:end_idx] = all_batch_loss.to(orig_device)

            # Compute final metrics
            robust_accuracy = (robust_flags.sum(dim=0) / n_examples).item()

            yaniv_accuracy = 100*(init_accuracy - robust_accuracy)/init_accuracy
            # Print verbose results if required
            if self.verbose:
                print("reporting results for adversarial attack: " + self.attack_name)
                print(f"clean accuracy: {init_accuracy:.2%}")
                print(f"robust accuracy (LOWER IS BETTER): {robust_accuracy:.2%}")
                print("perturbations max L_inf:", perts_max_l_inf)
                print("yaniv_accuracy (HIGHER IS BETTER):", yaniv_accuracy)

        return universal_pert, init_accuracy, x_adv, y_adv, robust_accuracy, adv_loss, None, None, perts_max_l_inf, None, None, None, None, yaniv_accuracy, mean_loss_per_step
