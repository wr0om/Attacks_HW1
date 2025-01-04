import torch
from attacks.pgd_attacks.attack import Attack


class UPGD(Attack):
    def __init__(
            self,
            model,
            criterion,
            misc_args=None,
            pgd_args=None):
        super(UPGD, self).__init__(model, criterion, misc_args, pgd_args)

    def report_schematics(self):

        print("Attack L_inf norm limitation:")
        print(self.eps_ratio)
        print("Number of iterations for perturbation optimization:")
        print(self.n_iter)
        print("Number of restarts for perturbation optimization:")
        print(self.n_restarts)

    def perturb(self, x, y, targeted=False):
        raise NotImplementedError('You need to define a perturb method!')
