Files Description:

run_attack.py:  This is the main file to run, which handles argument processing, loads the model, and deploys the attack accordingly. If instructed, it saves the resulting adversarial pertubations and inputs.

parser.py:  Argument processing. Details the available parameters and their default values.

AdvRunner.py: Receives Model, Attack, and Dateset and deploys the attack over the model for each sample in the Dataset.

attacks folder: Contains available attacks

attacks/pgd_attacks: Implementation of iterative FGSM (PGD) attacks.

attacks/pgd_attacks/attack.py: Base class for PGD attacks

attacks/pgd_attacks/pgd.py: Implementation of standard PGD attack

attacks/pgd_attacks/universal.py: Stump for implementing universal PGD attack

results folder: Resulting adversarial pertubations and inputs are saved here

models folder: Contains pre-trained models, downloaded via Robustbench

data folder: Contains Datasets, downloaded via Pytorch.
