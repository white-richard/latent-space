import matplotlib.pyplot as plt
import pacmap
import timm
import torch
import torchvision.transforms as transforms

from latent_space.imagenet_dataloader import get_imagenet_loaders

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

data_dir = "/home/richiewhite/.code/datasets/imagenet"

loaders = get_imagenet_loaders(data_dir, batch_size=32, num_workers=4, limit_data=True)
test_loader = loaders["val"]


model = timm.create_model("vit_base_patch16_224.dino", pretrained=True, num_classes=0)
model = model.to(device)
model.eval()

model.eval()
X = []
y = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        emb = model(images)
        X.append(emb.cpu())
        y.append(labels.cpu())

X = torch.cat(X, dim=0).numpy()
y = torch.cat(y, dim=0).numpy()

# initializing the pacmap instance
# Setting n_neighbors to "None" leads to an automatic choice shown below in "parameter" section
embedding = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0)

# fit the data (The index of transformed data corresponds to the index of the original data)
X_transformed = embedding.fit_transform(X, init="pca")

# visualize the embedding
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.scatter(X_transformed[:, 0], X_transformed[:, 1], cmap="Spectral", c=y, s=0.6)
fig.savefig("./scatter_plot.png")
