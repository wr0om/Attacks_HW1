import numpy as np
import torch
from tqdm import trange

# Class for running adversarial attacks and evaluations on a given model
class AdvRunner:
    def __init__(self, model, attack, data_RGB_size, device, dtype, verbose=False):
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

    # Evaluate the model on clean inputs
    def run_clean_evaluation(self, x_orig, y_orig, n_examples, orig_device):
        robust_flags = torch.zeros(n_examples, dtype=torch.bool, device=orig_device)

        # Transfer data to the device
        x = x_orig.clone().detach().to(self.device)
        y = y_orig.clone().detach().to(self.device)

        # Get model predictions
        output = self.model.forward(x)
        correct_batch = y.eq(output.max(dim=1)[1]).detach().to(orig_device)
        robust_flags = correct_batch

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
    def run_standard_evaluation(self, x_orig, y_orig, n_examples):
        with torch.no_grad():
            orig_device = x_orig.device

            # Evaluate on clean examples
            robust_flags, n_robust_examples, init_accuracy = self.run_clean_evaluation(
                x_orig, y_orig, n_examples, orig_device
            )

        # Generate universal adversarial perturbation
        with torch.cuda.device(self.device):
            x = x_orig.clone().detach().to(self.device)
            y = y_orig.clone().detach().to(self.device)

            # Generate the universal perturbation
            universal_pert, adv_loss = self.attack.perturb(x, y)

        # Evaluate the universal perturbation on the entire dataset
        with torch.no_grad():
            # Apply the universal perturbation to all samples
            x_adv = torch.clamp(x + universal_pert.unsqueeze(0), 0, 1)

            # Get predictions on adversarial examples
            output = self.model.forward(x_adv)
            y_adv = output.max(dim=1)[1].to(orig_device)  # Ensure y_adv is on orig_device

            # Ensure y is also on the same device
            y = y.to(orig_device)

            # Update robust flags for incorrect predictions
            false_batch = ~y.eq(y_adv).detach()
            robust_flags[false_batch] = False

            # Compute final metrics
            robust_accuracy = (robust_flags.sum(dim=0) / n_examples).item()
            perts_max_l_inf = universal_pert.abs().max().item()

            # Print verbose results if required
            if self.verbose:
                print("reporting results for adversarial attack: " + self.attack_name)
                print(f"clean accuracy: {init_accuracy:.2%}")
                print(f"robust accuracy: {robust_accuracy:.2%}")
                print("perturbations max L_inf:", perts_max_l_inf)

        
        return universal_pert, init_accuracy, x_adv, y_adv, robust_accuracy, adv_loss, None, None, perts_max_l_inf, None, None, None, None
