# config
device = 'cuda'
model_path = './model/model.bin'
emb_dim = 128
hidden_size = 256
vocab_size = 30
batch_size = 2
test_batch_size = 1
max_dec_steps = 10
adagrad_init_acc = 0.1
rand_unif_init_mag = 0.02
trunc_norm_init_std = 1e-4
lr = 0.15
eps = 1e-12
max_epochs = 3000
log_epochs = 1
