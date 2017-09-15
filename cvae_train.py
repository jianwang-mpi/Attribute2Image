import torch
import torch.optim as optim
import reader
import condVAE

def reparameter():
    pass

encoder = condVAE.CVAE_encoder(zdim=256, ydim=73)
decoder = condVAE.CVAE_decoder(zdim=256, ydim=73)

for imgs, attrs in reader.train_loader:
    pass