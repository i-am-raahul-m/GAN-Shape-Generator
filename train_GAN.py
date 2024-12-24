import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from math import ceil
import pickle


### Program Context
# Check if GPU availability
gpu_available = torch.cuda.is_available()
device = torch.device("cuda" if gpu_available else "cpu")
print(f"Using device: {device}")
epochs_per_train_stat = 100

# Paths to sources and destinations
path_to_dataset = "Datasets/"
dataset_of_choice = "circles/"  # circles, triangles, squares
model_save_path = "Models/"
model_num = 1
generator_save_file = f"GAN{model_num}-generator.pth"
discriminator_save_file = f"GAN{model_num}-discriminator.pth"
stats_save_path = "Training_Statistics/"
stats_save_file = f"GAN{model_num}-train_stats.pkl"


### HYPER-PARAMETERS
# CONSTANTS (for available datasets)
dataset_size = 100
img_height = 28
img_width = 28
data_dim = img_height * img_width

# VARIABLES
latent_dim = 100
learning_rate = 0.00001
batch_size = 10
epochs = 2000
num_of_batches = ceil(dataset_size / batch_size)


### GAN Architecture
# Generator Architecture (No Sigmoid)
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
            # Logits
        )

    def forward(self, z):
        return self.model(z)
    

# Discriminator Architecture (No Sigmoid)
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
            # Logits
        )
    
    def forward(self, x):
        return self.model(x)
    

### GAN Training Context
# Models
generator = Generator(input_dim=latent_dim, output_dim=data_dim).to(device)
discriminator = Discriminator(input_dim=data_dim).to(device)

# Loss Function: Using BCEWithLogitsLoss
loss_function = nn.BCEWithLogitsLoss()

# Optimizers
optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)


### Input tensor creation
# Noise tensor input to generator
def sample_noise(batch_size, latent_dim):
    return torch.randn(batch_size, latent_dim, device=device)

# Dataset samples tensor input to discriminator
def sample_real_data(batch_size, batch_idx):
    array = np.zeros((batch_size, data_dim), dtype=np.float32)
    for i in range(1, batch_size + 1):
        img = Image.open(path_to_dataset + dataset_of_choice + f"drawing({batch_idx * batch_size + i}).png")
        img_arr = np.array(img, dtype=np.float32).flatten()
        img_arr /= 255
        array[i - 1] = img_arr

    return torch.tensor(array, dtype=torch.float32, device=device)


### Training Loop
g_loss_history = np.zeros(epochs // epochs_per_train_stat)
d_loss_history = np.zeros(epochs // epochs_per_train_stat)

for epoch in range(epochs):
    for batch_idx in range(num_of_batches):

        # Training Discriminator
        real_data = sample_real_data(batch_size, batch_idx)
        fake_data = generator(sample_noise(batch_size, latent_dim)).detach()

        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        optimizer_d.zero_grad()
        real_loss = loss_function(discriminator(real_data), real_labels)
        fake_loss = loss_function(discriminator(fake_data), fake_labels)
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_d.step()

        # Training Generator
        fake_data = generator(sample_noise(batch_size, latent_dim))
        fake_labels = torch.ones(batch_size, 1, device=device)

        optimizer_g.zero_grad()
        g_loss = loss_function(discriminator(fake_data), fake_labels)
        g_loss.backward()
        optimizer_g.step()

    if epoch % epochs_per_train_stat == 0:
        g_loss_val = g_loss.item()
        d_loss_val = d_loss.item()
        print(f"Completed: {epoch}/{epochs}  G Loss: {g_loss_val:.4f}  D Loss: {d_loss_val:.4f}")
        g_loss_history[epoch // epochs_per_train_stat] = g_loss_val
        d_loss_history[epoch // epochs_per_train_stat] = d_loss_val



### Saving trained models
# Save trained parameters of model components
torch.save(generator.state_dict(), model_save_path + generator_save_file)
torch.save(discriminator.state_dict(), model_save_path + discriminator_save_file)

# Save training statistics
with open(stats_save_path + stats_save_file, "wb") as file:
    pickle.dump(g_loss_history, file)
    pickle.dump(d_loss_history, file)
