import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skimage.transform import radon, iradon
from keras.models import Sequential
from keras.layers import Dense, Reshape, Conv2DTranspose, Conv2D, Flatten
from sklearn.metrics import mean_squared_error

# Load MNIST dataset
from keras.datasets import mnist

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize pixel values to range [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Split dataset into training and testing sets
X_train, X_val = train_test_split(X_train, test_size=0.2, random_state=42)


# Function to compute sinograms for a given set of images
def compute_sinograms(images, angles):
    sinograms = []
    for image in images:
        sinogram = radon(image, theta=angles, circle=False)
        sinograms.append(sinogram)
    return np.array(sinograms)


def calculate_snr(original, reconstructed):
    signal_power = np.sum(original ** 2)
    noise_power = np.sum((original - reconstructed) ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


# Add Gaussian noise to sinograms
def add_gaussian_noise(sinograms, noise_level):
    noisy_sinograms = []
    for sinogram in sinograms:
        noise = np.random.normal(0, noise_level, sinogram.shape)
        noisy_sinogram = sinogram + noise
        noisy_sinograms.append(noisy_sinogram)
    return np.array(noisy_sinograms)


# Compute sinograms for training and testing images
# angles = np.linspace(0., 180., max(X_train.shape[1], X_train.shape[2]), endpoint=False) # 28 view angles
# angles = np.array([0, 45, 90, 135])
angles = np.linspace(0, 180, 10, endpoint=False)  # 10 view angles
train_sinograms = compute_sinograms(X_train, angles)
val_sinograms = compute_sinograms(X_val, angles)
test_sinograms = compute_sinograms(X_test, angles)

# Define AUTOMAP model
model = Sequential([
    Reshape((train_sinograms.shape[1], train_sinograms.shape[2], 1),
            input_shape=(train_sinograms.shape[1], train_sinograms.shape[2])),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2DTranspose(1, (3, 3), activation='relu', padding='same'),
    Flatten(),
    Dense(np.prod(X_train.shape[1:]), activation='sigmoid'),
    Reshape(X_train.shape[1:])
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(train_sinograms, X_train, epochs=30, batch_size=32,
                    validation_data=(val_sinograms, X_val))

# Extract loss and validation loss from the history object
loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot loss curve
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_validation_loss.png')
plt.show()

# Reconstruct images for testing set
reconstructed_images = model.predict(test_sinograms)

# Reconstruct projections of testing images using FBP algorithm
fbp_reconstructions = []
for sinogram in test_sinograms:
    fbp_reconstruction = iradon(sinogram, theta=angles, circle=False)
    fbp_reconstructions.append(fbp_reconstruction)
fbp_reconstructions = np.array(fbp_reconstructions)

rmse_automap = [np.sqrt(mean_squared_error(X_test[i], reconstructed_images[i])) for i in range(len(X_test))]
rmse_fbp = [np.sqrt(mean_squared_error(X_test[i], fbp_reconstructions[i])) for i in range(len(X_test))]

snr_automap = [calculate_snr(X_test[i], reconstructed_images[i]) for i in range(len(X_test))]
snr_fbp = [calculate_snr(X_test[i], fbp_reconstructions[i]) for i in range(len(X_test))]

# Show and save comparison of original, AUTOMAP, and FBP reconstructed images
n = 4
plt.figure(figsize=(15, 6))
for i in range(n):
    # Original images
    plt.subplot(3, n, i + 1)
    plt.imshow(X_test[i], cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')

    # Reconstructed images using AUTOMAP
    plt.subplot(3, n, i + 1 + n)
    plt.imshow(reconstructed_images[i], cmap='gray')
    plt.title('AUTOMAP\nRMSE: {:.4f}\nSNR: {:.2f} dB'.format(rmse_automap[i], snr_automap[i]))
    plt.axis('off')

    # Reconstructed images using FBP
    plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(fbp_reconstructions[i], cmap='gray')
    plt.title('FBP\nRMSE: {:.4f}\nSNR: {:.2f} dB'.format(rmse_fbp[i], snr_fbp[i]))
    plt.axis('off')

plt.tight_layout()
plt.savefig('reconstructed_images_comparison.png')
plt.show()

# Calculate and show differences between AUTOMAP and FBP reconstructions
differences = np.abs(reconstructed_images - fbp_reconstructions)
print("Maximum absolute difference between AUTOMAP and FBP reconstructions:", np.max(differences))

# Plot RMSE values separately
plt.plot(rmse_automap, marker='o', label='AUTOMAP')
plt.plot(rmse_fbp, marker='+', label='FBP')
plt.title('RMSE of AUTOMAP vs FBP')
plt.xlabel('Image Id')
plt.ylabel('RMSE')
plt.legend()
plt.xticks(range(n), range(1, n + 1))
plt.tight_layout()
plt.savefig('RMSE.png')
plt.show()

# Plot SNR values separately
plt.plot(snr_automap, marker='o', label='AUTOMAP')
plt.plot(snr_fbp, marker='+', label='FBP')
plt.title('SNR of AUTOMAP vs FBP')
plt.xlabel('Image Id')
plt.ylabel('SNR')
plt.legend()
plt.xticks(range(n), range(1, n + 1))
plt.tight_layout()
plt.savefig('SNR.png')
plt.show()

# Test model robustness in noise
train_sinograms_noisy = add_gaussian_noise(train_sinograms, noise_level=0.2)
val_sinograms_noisy = add_gaussian_noise(val_sinograms, noise_level=0.2)
test_sinograms_noisy = add_gaussian_noise(test_sinograms, noise_level=0.2)

model_noisy = Sequential([
    Reshape((train_sinograms.shape[1], train_sinograms.shape[2], 1),
            input_shape=(train_sinograms.shape[1], train_sinograms.shape[2])),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2DTranspose(1, (3, 3), activation='relu', padding='same'),
    Flatten(),
    Dense(np.prod(X_train.shape[1:]), activation='sigmoid'),
    Reshape(X_train.shape[1:])
])

# Compile the model
model_noisy.compile(optimizer='adam', loss='mse')

# Train the model with noisy sinograms
history_noisy = model_noisy.fit(train_sinograms, X_train, epochs=30, batch_size=32,
                                validation_data=(val_sinograms, X_val))

# Extract loss and validation loss from the history object
loss_noisy = history_noisy.history['loss']
val_loss_noisy = history_noisy.history['val_loss']

# Plot loss curve
plt.plot(loss_noisy, label='Noisy Training Loss')
plt.plot(val_loss_noisy, label='Noisy Validation Loss')
plt.title('Noisy Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Noisy Loss')
plt.legend()
plt.savefig('training_validation_loss_noisy.png')
plt.show()

reconstructed_images_noisy = model_noisy.predict(test_sinograms_noisy)

fbp_reconstructions_noisy = []
for sinogram in test_sinograms_noisy:
    fbp_reconstruction_noisy = iradon(sinogram, theta=angles, circle=False)
    fbp_reconstructions_noisy.append(fbp_reconstruction_noisy)
fbp_reconstructions_noisy = np.array(fbp_reconstructions_noisy)

rmse_automap_noisy = [np.sqrt(mean_squared_error(X_test[i], reconstructed_images_noisy[i])) for i in range(len(X_test))]
rmse_fbp_noisy = [np.sqrt(mean_squared_error(X_test[i], fbp_reconstructions_noisy[i])) for i in range(len(X_test))]
snr_automap_noisy = [calculate_snr(X_test[i], reconstructed_images_noisy[i]) for i in range(len(X_test))]
snr_fbp_noisy = [calculate_snr(X_test[i], fbp_reconstructions_noisy[i]) for i in range(len(X_test))]

n = 4
id = np.random.choice(len(X_test), n, replace=False)
plt.figure(figsize=(15, 6))
for i in range(n):
    plt.subplot(n, 5, 1 + 5 * i)
    plt.imshow(X_test[id[i]], cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')

    plt.subplot(n, 5, 2 + 5 * i)
    plt.imshow(reconstructed_images[id[i]], cmap='gray')
    plt.title('AUTOMAP\nRMSE: {:.4f}\nSNR: {:.2f} dB'.format(rmse_automap[id[i]], snr_automap[id[i]]))
    plt.axis('off')

    plt.subplot(n, 5, 3 + 5 * i)
    plt.imshow(reconstructed_images_noisy[id[i]], cmap='gray')
    plt.title('AUTOMAP-Noisy\nRMSE: {:.4f}\nSNR: {:.2f} dB'.format(rmse_automap_noisy[id[i]], snr_automap_noisy[id[i]]))
    plt.axis('off')

    plt.subplot(n, 5, 4 + 5 * i)
    plt.imshow(fbp_reconstructions[id[i]], cmap='gray')
    plt.title('FBP\nRMSE: {:.4f}\nSNR: {:.2f} dB'.format(rmse_fbp[id[i]], snr_fbp[id[i]]))
    plt.axis('off')

    plt.subplot(n, 5, 5 + 5 * i)
    plt.imshow(fbp_reconstructions_noisy[id[i]], cmap='gray')
    plt.title('FBP-Noisy\nRMSE: {:.4f}\nSNR: {:.2f} dB'.format(rmse_fbp_noisy[id[i]], snr_fbp_noisy[id[i]]))
    plt.axis('off')

plt.tight_layout()
plt.savefig('reconstructed_images_noisy_comparison.png')
plt.show()
