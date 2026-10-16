# train.py - training and saving the baseline GAN

# TODO: imports (torch, torch.nn.functional, time, matplotlib, your dataset.py + modules.py)


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------
# TODO: epochs, learning rate, batch size, hidden size, number of GRU layers, random seed
# TODO: device: the GPU if there is one, otherwise the CPU


# ---------------------------------------------------------------------------
# training loop
# ---------------------------------------------------------------------------
# TODO: def train_baseline(model, loader, epochs, lr, device)
# TODO:   one Adam optimiser for the generator, one for the discriminator
# TODO:   for each epoch, for each batch of real windows:
# TODO:     move the batch to the device
# TODO:     make the same number of fake windows from fresh noise
#
# TODO:     DISCRIMINATOR step
# TODO:       score the real windows and the fake ones
# TODO:       loss: real should score "real", fake should score "fake" (binary cross-entropy)
# TODO:       think: should gradients flow back into the generator here?
# TODO:       zero grads -> backward -> step
#
# TODO:     GENERATOR step
# TODO:       score fresh fake windows
# TODO:       loss: the generator wins when fake scores "real"
# TODO:       zero grads -> backward -> step
#
# TODO:   average both losses over the epoch, print them, keep them in a list
# TODO:   return the loss history


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
# TODO: if __name__ == "__main__":
# TODO:   set the seed (so runs can be repeated)
# TODO:   get the loaders + mean + std from dataset.py
# TODO:   build the BaselineGAN, move it to the device, print its parameter count
# TODO:   time the training (start / end)
# TODO:   train
# TODO:   save the model weights + mean + std + settings in one file (not committed - .gitignore!)
# TODO:   plot generator and discriminator loss per epoch, save as a png
# TODO:   print training time; peak GPU memory if on GPU   (acceptance criterion 5)
# TODO:   smoke test first: 1 epoch, a few batches - does it run end to end?


# LATER: TimeGAN's three phases (autoencoder -> supervisor -> joint)
