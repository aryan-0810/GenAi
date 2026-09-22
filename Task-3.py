# GAN Assignment

import numpy as np

# 1. Simple Generator

def simple_generator(random_vector):
    np.random.seed(int(np.sum(random_vector) * 1000) % 100000)
    image = np.random.rand(28, 28)
    return image


random_vector = np.random.rand(100)

fake_image = simple_generator(random_vector)

print("1. SIMPLE GENERATOR")
print("-" * 40)
print("Generated image shape:", fake_image.shape)
print("Minimum pixel value:", fake_image.min())
print("Maximum pixel value:", fake_image.max())
print()

# 2. Discriminator

class Discriminator:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def discriminate(self, image):
        mean_value = np.mean(image)

        if mean_value >= self.threshold:
            return 1.0
        else:
            return 0.0


discriminator = Discriminator()

score = discriminator.discriminate(fake_image)

print("2. DISCRIMINATOR")
print("-" * 40)
print("Discriminator score:", score)

if score == 1.0:
    print("Prediction: Real")
else:
    print("Prediction: Fake")

print()

# 3. Adversarial Training Cycle

random_vector = np.random.rand(100)

fake_image = simple_generator(random_vector)

score = discriminator.discriminate(fake_image)

actual_label = 0

print("3. ADVERSARIAL TRAINING CYCLE")
print("-" * 40)
print("Fake image generated successfully.")
print("Discriminator score:", score)

if score == actual_label:
    print("Discriminator correctly identified the fake image.")
else:
    print("Discriminator incorrectly identified the fake image.")

print()

# 4. Real-World Applications of GANs

applications = {
    "Adobe Photoshop": "GAN-based technology can be used for image editing and generative features by creating or modifying realistic visual content.",
    "NVIDIA StyleGAN Applications": "StyleGAN can generate highly realistic images and can also be used for image manipulation and creative visual content generation."
}

print("4. REAL-WORLD GAN APPLICATIONS")
print("-" * 40)

for application, description in applications.items():
    print(application + ":")
    print(description)
    print()

# 5. Difference Between DCGAN, StyleGAN and CycleGAN

prompt = """
Explain the difference between DCGAN, StyleGAN, and CycleGAN in simple language
in 3 to 4 lines. Explain what each GAN is mainly used for and keep the explanation
easy for a Data Science student to understand.
"""

response = """
DCGAN is a GAN architecture that uses convolutional neural networks to generate
realistic images and is commonly used for basic image generation tasks.

StyleGAN is an advanced GAN architecture that provides better control over the
style and different visual features of generated images.

CycleGAN is mainly used for image-to-image translation, such as converting
images from one visual domain to another without requiring paired training images.
"""

learned = """
I learned that CycleGAN can perform image-to-image translation without needing
paired images, while StyleGAN provides more control over different visual features
of generated images.
"""

print("5. DIFFERENCE BETWEEN DCGAN, STYLEGAN AND CYCLEGAN")
print("-" * 50)

print("PROMPT USED:")
print(prompt)

print("CHATGPT RESPONSE:")
print(response)

print("ONE THING I LEARNED:")
print(learned)
