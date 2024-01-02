import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from prepare_dataset import prepare_dataset
from diffusion_model import SimpleUnet, DiffusionProcess
import matplotlib.pyplot as plt

# Set random seed for reproducibility
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

dataroot = "data_faces/img_align_celeba" # Root directory for dataset
batch_size = 128 # Batch size during training
image_size = 64 # image size

T = 300 # Number of time steps
num_epochs = 5 # Number of training epochs
lr = 0.001 # Learning rate for optimizers

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
dataloader = prepare_dataset(root=dataroot, device=device, image_size=image_size, batch_size=batch_size)

# Create the Unet
model = SimpleUnet().to(device)
print("Num params: ", sum(p.numel() for p in model.parameters()))
optimizer = optim.Adam(model.parameters(), lr=lr)
diff_process = DiffusionProcess(T=T, img_size=image_size, device=device).to(device)

losses = []
for epoch in range(num_epochs):
    for i, batch in enumerate(dataloader):
        if batch.shape[0] != batch_size:
            continue
        optimizer.zero_grad()

        t = torch.randint(0, T, (batch_size,), device=device).long()
        batch.to(device)
        loss = diff_process.get_loss(model, batch, t)
        loss.backward()
        optimizer.step()
        losses.append(loss.mean().item())

        if i % 50 == 0:
            print('[%d/%d][%d/%d] Loss: %.4f' % 
                  (epoch, num_epochs, i, len(dataloader), loss.mean().item()))

    diff_process.sample_plot_image(model=model,
                                   save_path=f"output/diffusion_sample_{epoch}.png")

# save loss as image
plt.figure(figsize=(10,5))
plt.title("Diffusion Loss During Training")
plt.plot(losses)
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.savefig('diffusion_loss.png')

# save model
torch.save(model, 'models/diffusion_model.pth')
