# Autoencoder and Variational Autoencoder Assignment

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.datasets import fashion_mnist

(x_train, _), (x_test, _) = fashion_mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# 1. Simple Autoencoder using MSE

input_layer = layers.Input(shape=(784,))

encoded = layers.Dense(128, activation="relu")(input_layer)
encoded = layers.Dense(64, activation="relu")(encoded)
encoded = layers.Dense(32, activation="relu")(encoded)

decoded = layers.Dense(64, activation="relu")(encoded)
decoded = layers.Dense(128, activation="relu")(decoded)
decoded = layers.Dense(784, activation="sigmoid")(decoded)

autoencoder_mse = Model(input_layer, decoded)

autoencoder_mse.compile(
optimizer="adam",
loss="mse"
)

autoencoder_mse.fit(
x_train,
x_train,
epochs=5,
batch_size=256,
validation_data=(x_test, x_test)
)

reconstructed = autoencoder_mse.predict(x_test[:1])

plt.figure(figsize=(6, 3))

plt.subplot(1, 2, 1)
plt.imshow(x_test[0].reshape(28, 28), cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(reconstructed[0].reshape(28, 28), cmap="gray")
plt.title("Reconstructed")
plt.axis("off")

plt.show()

# 2. Autoencoder using Binary Crossentropy

input_layer_bce = layers.Input(shape=(784,))

encoded_bce = layers.Dense(128, activation="relu")(input_layer_bce)
encoded_bce = layers.Dense(64, activation="relu")(encoded_bce)
encoded_bce = layers.Dense(32, activation="relu")(encoded_bce)

decoded_bce = layers.Dense(64, activation="relu")(encoded_bce)
decoded_bce = layers.Dense(128, activation="relu")(decoded_bce)
decoded_bce = layers.Dense(784, activation="sigmoid")(decoded_bce)

autoencoder_bce = Model(input_layer_bce, decoded_bce)

autoencoder_bce.compile(
loss="binary_crossentropy",
optimizer="adam"
)

autoencoder_bce.fit(
x_train,
x_train,
epochs=5,
batch_size=256,
validation_data=(x_test, x_test)
)

mse_images = autoencoder_mse.predict(x_test[:3])
bce_images = autoencoder_bce.predict(x_test[:3])

plt.figure(figsize=(9, 6))

for i in range(3):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(3, 3, i + 4)
    plt.imshow(mse_images[i].reshape(28, 28), cmap="gray")
    plt.title("MSE")
    plt.axis("off")

    plt.subplot(3, 3, i + 7)
    plt.imshow(bce_images[i].reshape(28, 28), cmap="gray")
    plt.title("BCE")
    plt.axis("off")

plt.tight_layout()
plt.show()

# 3. Image Denoising

noise_factor = 0.4

noisy_images = x_test[:10] + noise_factor * np.random.normal(
    loc=0.0,
    scale=1.0,
    size=x_test[:10].shape
)

noisy_images = np.clip(noisy_images, 0.0, 1.0)

denoised_images = autoencoder_bce.predict(noisy_images)

plt.figure(figsize=(12, 6))

for i in range(10):
    plt.subplot(2, 10, i + 1)
    plt.imshow(noisy_images[i].reshape(28, 28), cmap="gray")
    plt.title("Noisy")
    plt.axis("off")

    plt.subplot(2, 10, i + 11)
    plt.imshow(denoised_images[i].reshape(28, 28), cmap="gray")
    plt.title("Denoised")
    plt.axis("off")

plt.tight_layout()
plt.show()

# 4. Variational Autoencoder

latent_dim = 2

encoder_inputs = layers.Input(shape=(784,))

x = layers.Dense(128, activation="relu")(encoder_inputs)
x = layers.Dense(64, activation="relu")(x)

z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(
        shape=tf.shape(z_mean)
    )
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_var])

encoder = Model(
    encoder_inputs,
    [z_mean, z_log_var, z]
)

latent_inputs = layers.Input(shape=(latent_dim,))

x = layers.Dense(64, activation="relu")(latent_inputs)
x = layers.Dense(128, activation="relu")(x)
decoder_outputs = layers.Dense(
    784,
    activation="sigmoid"
)(x)

decoder = Model(
    latent_inputs,
    decoder_outputs
)

class VAE(Model):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):

        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:

            z_mean, z_log_var, z = self.encoder(data)

            reconstruction = self.decoder(z)

            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(
                    data,
                    reconstruction
                )
            )

            kl_loss = -0.5 * tf.reduce_mean(
                1 + z_log_var
                - tf.square(z_mean)
                - tf.exp(z_log_var)
            )

            total_loss = reconstruction_loss + kl_loss

        gradients = tape.gradient(
            total_loss,
            self.trainable_weights
        )

        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_weights)
        )

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss
        }

vae = VAE(encoder, decoder)

vae.compile(
    optimizer="adam"
)

vae.fit(
    x_train,
    epochs=5,
    batch_size=256
)

random_latent_points = np.random.normal(
    size=(5, latent_dim)
)

generated_images = decoder.predict(
    random_latent_points
)

plt.figure(figsize=(10, 2))

for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(
        generated_images[i].reshape(28, 28),
        cmap="gray"
    )
    plt.title("Generated")
    plt.axis("off")

plt.tight_layout()
plt.show()

# 5. KL-Divergence Explanation

prompt = """
Explain in simple words what KL-divergence loss does in a Variational Autoencoder
and why it is needed. Explain it from the perspective of a beginner learning
Generative AI and include a simple example.
"""

response = """
KL-divergence loss is used in a Variational Autoencoder to keep the latent space
organized and close to a standard normal distribution.

A VAE has an encoder that converts an input image into a probability distribution
represented by a mean and variance. Instead of allowing the encoder to create
completely random distributions for different images, KL-divergence adds a penalty
when these distributions move too far away from the standard normal distribution.

This is important because it makes the latent space smooth and continuous. Similar
images can have similar locations in the latent space, and we can sample random
points from this space and pass them through the decoder to generate new images.

For example, if the latent space is well organized, images of similar clothing
items can be located close to each other. When we randomly select a point between
these areas, the decoder can generate a meaningful new image.

Therefore, the VAE loss has two important parts: reconstruction loss makes sure
the decoded image is similar to the original image, while KL-divergence loss keep
the latent space organized.
"""