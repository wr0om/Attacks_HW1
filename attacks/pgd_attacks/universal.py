import torch
from attacks.pgd_attacks.attack import Attack
from torch.utils.data import TensorDataset, DataLoader



class UPGD(Attack):
    def __init__(
            self,
            model,
            criterion,
            misc_args=None,
            pgd_args=None):
        super(UPGD, self).__init__(model, criterion, misc_args, pgd_args)
        self.pert = None
    # def random_initialization(self, single=True):
    #     wanted_shape = self.data_shape
    #     if single:
    #         wanted_shape = [1] + self.data_shape[1:]
    #     if self.norm == 'Linf':
    #         return torch.empty(wanted_shape, dtype=self.dtype, device=self.device).uniform_(-1, 1) * self.eps
    #     else:
    #         return torch.empty(wanted_shape, dtype=self.dtype, device=self.device).normal_(0, self.eps * self.eps)
        
    # def set_params(self, x, targeted):
    #     self.batch_size = x.shape[0]
    #     self.data_shape[0] = x.shape[0]
    #     self.set_multiplier(targeted)
    #     self.pert_lb = self.data_RGB_start - x
    #     self.pert_ub = self.data_RGB_end - x
    #     # we only use a single perturbation for all images in the batch
    #     self.pert_lb = torch.min(self.pert_lb, dim=0, keepdim=True)[0]
    #     self.pert_ub = torch.max(self.pert_ub, dim=0, keepdim=True)[0]

    def report_schematics(self):

        print("Attack L_inf norm limitation:")
        print(self.eps_ratio)
        print("Number of iterations for perturbation optimization:")
        print(self.n_iter)
        print("Number of restarts for perturbation optimization:")
        print(self.n_restarts)

    def perturb(self, x, y, targeted=False):
        """
            x.shape: torch.Size([250, 3, 32, 32])
            y.shape: torch.Size([250])
        """
        # SIMPLE UPGD (calculate perturbation for each image separately)

        with torch.no_grad():
            self.set_params(x, targeted)
            self.clean_loss, self.clean_succ = self.eval_pert(x, y, pert=torch.zeros_like(x))
            best_pert = torch.zeros_like(x)
            best_loss = self.clean_loss.clone().detach()
            best_succ = self.clean_succ.clone().detach()

            if self.report_info:
                all_best_succ = torch.zeros(self.n_restarts,  self.n_iter + 1, self.batch_size, dtype=torch.bool, device=self.device)
                all_best_loss = torch.zeros(self.n_restarts,  self.n_iter + 1, self.batch_size, dtype=self.dtype, device=self.device)
            else:
                all_best_succ = None
                all_best_loss = None

        self.model.eval()
        for rest in range(self.n_restarts):
            if not self.pert:
                if self.rand_init:
                    pert_init = self.random_initialization()
                    pert_init = self.project(pert_init)
                else:
                    pert_init = torch.zeros_like(x)
            else:
                pert_init = self.pert
            with torch.no_grad():
                loss, succ = self.eval_pert(x, y, pert_init)
                self.update_best(best_loss, loss, [best_pert, best_succ], [pert_init, succ])
                if self.report_info:
                    all_best_succ[rest, 0] = best_succ
                    all_best_loss[rest, 0] = best_loss

            pert = pert_init.clone().detach()
            for k in range(1, self.n_iter + 1):
                temp_pert = pert[0]
                temp_pert.requires_grad_()
                pert = temp_pert.repeat(self.batch_size, 1, 1, 1)
                train_loss = self.criterion(self.model.forward(x+pert), y)
                grad = torch.autograd.grad(train_loss.mean(), [temp_pert])[0].detach()
                temp_pert = temp_pert.unsqueeze(0)
                with torch.no_grad():
                    grad = self.normalize_grad(grad)
                    temp_pert += self.multiplier * grad
                pert = temp_pert.repeat(self.batch_size, 1, 1, 1)
                pert = self.project(pert)
                loss, succ = self.eval_pert(x, y, pert)
                self.update_best(best_loss, loss, [best_pert, best_succ], [pert, succ])
                if self.report_info:
                    all_best_succ[rest, k] = succ
                    all_best_loss[rest, k] = loss

        adv_pert = best_pert.clone().detach()
        adv_pert_loss = best_loss.clone().detach()
        self.pert = adv_pert
        return adv_pert, adv_pert_loss, all_best_succ, all_best_loss

