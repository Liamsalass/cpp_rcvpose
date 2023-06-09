import torch

folder_path = r'C:\Users\User\.cw\work\cpp_rcvpose\rcvpose\out\build\x64-debug\train_kpt1\model_best'

# Load the model information
model_info = torch.jit.load(folder_path + '/info.pt')

# Extract the relevant information
epoch = model_info.epoch
iteration = model_info.iteration
model_arch = model_info.arch
best_acc_mean = model_info.best_acc_mean
val_loss = model_info.loss
optimizer = model_info.optimizer
current_lr = model_info.lr

# Load the model itself
model = torch.nn.ModuleDict(folder_path + '/model.pt')

# Set the model to evaluation mode
#model.eval()

print("epoch, ", epoch)
print("iteration, ", iteration)
print("model_arch, ", model_arch)
print("best_acc_mean, ", best_acc_mean)
print("val_loss, ", val_loss)
print("optimizer, ", optimizer)
print("current_lr, ", current_lr)
