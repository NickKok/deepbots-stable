import PPORun

"""
Modify these constants if needed.
""" 
# The project name on the wandb
PROJECT_NAME = "Test"
WANDB_SAVE_FREQ = 400 # Logging Frequency on wandb
TRAIN = True # Train or evaluate the model
USE_Ray = False # Use Ray or Stable-Baseline
OUTPUT  = 'results/' # Output dir
Trained_model = 'path-of-model'

"""
Both model hyperparameters (Ray, Stable-Baseline), modify if needed.
""" 
lr = 1e-4
gamma = 0.995
gae_lambda =  0.95
clip_param =  0.2

"""
Ray model hyperparameters modify if needed.
""" 
kl_coeff = 1.0
num_sgd_iter = 10
sgd_minibatch_size = 64
train_batch_size = 64
num_workers = 1
num_gpus = 0
batch_mode = "complete_episodes"
observation_filter ="MeanStdFilter"

kwargs = {"lr":lr, "gamma":gamma, "gae_lambda":gae_lambda, 
            "clip_param":clip_param, "kl_coeff":kl_coeff, 
            "num_sgd_iter":num_sgd_iter, 
            "sgd_minibatch_size":sgd_minibatch_size,
            "train_batch_size":train_batch_size, 
            "num_workers":num_workers, "num_gpus":num_gpus,
            "batch_mode":batch_mode, 
            "observation_filter":observation_filter}
   
if __name__ == '__main__':
    print("Hello")
  
    PPORun.run(PROJECT_NAME, WANDB_SAVE_FREQ, TRAIN, Trained_model, USE_Ray,
               OUTPUT, **kwargs)
