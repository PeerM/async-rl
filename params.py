import logging

# Training Parameters
steps = 8000000 # Number of steps to run the training.
t_max = 5 # Number of steps between training.
beta = 1e-2 # Scale for the amount entropy regularizes the magnitude of the gradients.
lr = 7e-4 # Starting learning rate.
weight_decay = 0.0 # Do we decay weights with time?
gamma = 0.99 # Discount factor for future value estimates.

# Optimizer
RMSprop_lr = 7e-4 # Learning rate for the RMSProp optimization.
RMSprop_epsilon = 1e-1 # Value for epsilon in the RMSProp formula.
RMSprop_alpha = 0.99 # Value for alpha in the RMSProp formula.

# Network Parameters
use_ltsm = True # Whether to use a LTSM or a FF network before the final policy and value layers.

# Evaluation
eval_frequency = 100000 # How often we evaluate the model.
eval_n_runs = 5 # How many episodes to evaluate.

# Bookkeeping
log_level = logging.DEBUG
num_processes = 2
seed = None
outdir = "output"
