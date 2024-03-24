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


# Compute sinograms for training and testing images
# angles = np.linspace(0., 180., max(X_train.shape[1], X_train.shape[2]), endpoint=False) # 28 view angles
# angles = np.array([0, 45, 90, 135])
angles = np.linspace(0, 180, 10, endpoint=False)  # 10 view angles
train_sinograms = compute_sinograms(X_train, angles)
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
                    validation_data=(compute_sinograms(X_val, angles), X_val))

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
