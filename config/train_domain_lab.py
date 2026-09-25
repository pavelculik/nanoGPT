# Small starting configuration for one NVIDIA GPU. Run from nanoGPT root.
out_dir = 'out-domain-lab-01'
dataset = 'domain_lab'
init_from = 'scratch'
device = 'cuda'
compile = False
wandb_log = False

batch_size = 4
block_size = 128
gradient_accumulation_steps = 4
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.1

max_iters = 1000
eval_interval = 100
eval_iters = 20
log_interval = 10
always_save_checkpoint = True
learning_rate = 0.0003
min_lr = 0.00003
warmup_iters = 50
lr_decay_iters = 1000
# Keep the repository's automatic bfloat16 / float16 GPU selection.
