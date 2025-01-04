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

    def update_best(self, best_crit, new_crit, best_ls, new_ls):
        # because we only have a single perturbation for all images in the batch
        # we only need to compare the avg loss of the batch to decide if we should update the whole batch
        if new_crit.mean() > best_crit.mean():
            best_crit = new_crit.clone().detach()
            best_ls[0] = new_ls[0].clone().detach()
            best_ls[1] = new_ls[1].clone().detach()

        return best_ls[0], best_ls[1], best_crit

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
            if self.pert == None:
                if self.rand_init:
                    pert_init = self.random_initialization()
                    pert_init = self.project(pert_init)
                else:
                    pert_init = torch.zeros_like(x)
            else:
                pert_init = self.pert

            temp_pert = pert_init[0] # [3, 32, 32]
            #pert_init = temp_pert.repeat(self.batch_size, 1, 1, 1)
            pert_init = torch.stack([temp_pert]*self.batch_size, dim=0)
            print(f"1: {torch.unique(pert_init, dim=0).shape[0]}")
            with torch.no_grad():
                loss, succ = self.eval_pert(x, y, pert_init)
                old_best_pert = best_pert.clone().detach()
                best_pert, best_succ, best_loss = \
                    self.update_best(best_loss, loss, [best_pert, best_succ], [pert_init, succ])
                print(f"2: {torch.unique(best_pert, dim=0).shape[0]}")
                if self.report_info:
                    all_best_succ[rest, 0] = best_succ
                    all_best_loss[rest, 0] = best_loss

            pert = pert_init.clone().detach()
            for k in range(1, self.n_iter + 1):
                temp_pert = pert[0]
                temp_pert.requires_grad_()
                pert = temp_pert.repeat(self.batch_size, 1, 1, 1)
                print(f"3: {torch.unique(pert, dim=0).shape[0]}")
                train_loss = self.criterion(self.model.forward(x+pert), y)
                grad = torch.autograd.grad(train_loss.mean(), [temp_pert])[0].detach()
                temp_pert = temp_pert.unsqueeze(0)
                with torch.no_grad():
                    grad = self.normalize_grad(grad)
                    temp_pert += self.multiplier * grad

                #pert = temp_pert.repeat(self.batch_size, 1, 1, 1)
                pert = torch.stack([temp_pert[0]]*self.batch_size, dim=0)
                pert = self.project(pert)
                temp_pert = pert[0]
                pert = torch.stack([temp_pert]*self.batch_size, dim=0)

                print(f"4: {torch.unique(pert, dim=0).shape[0]}")
                loss, succ = self.eval_pert(x, y, pert)
                best_pert, best_succ, best_loss = \
                    self.update_best(best_loss, loss, [best_pert, best_succ], [pert, succ])
                print(f"5: {torch.unique(best_pert, dim=0).shape[0]}")
                if self.report_info:
                    all_best_succ[rest, k] = succ
                    all_best_loss[rest, k] = loss

        adv_pert = best_pert.clone().detach()
        adv_pert_loss = best_loss.clone().detach()
        self.pert = adv_pert
        print(f"6: {torch.unique(adv_pert, dim=0).shape[0]}")
        return adv_pert, adv_pert_loss, all_best_succ, all_best_loss

