# Diffusion Models Assignment

import cv2
import numpy as np
import os

# 1. Forward Diffusion Process

def add_noise(image, steps=10):
    noisy_images = []
    image = image.astype(np.float32) / 255.0

    for step in range(1, steps + 1):
        noise_level = step * 0.05
        noise = np.random.normal(0, noise_level, image.shape)

        noisy_image = image + noise
        noisy_image = np.clip(noisy_image, 0, 1)
        noisy_image = (noisy_image * 255).astype(np.uint8)

        noisy_images.append(noisy_image)

    return noisy_images


image = cv2.imread("input_image.png", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found. Please place input_image.png in the same folder.")
else:
    noisy_images = add_noise(image, 10)

    os.makedirs("diffusion_steps", exist_ok=True)

    for i, noisy_image in enumerate(noisy_images, start=1):
        filename = f"diffusion_steps/noisy_step_{i}.png"
        cv2.imwrite(filename, noisy_image)

    print("1. FORWARD DIFFUSION")
    print("-" * 40)
    print("10 noisy images have been generated and saved.")
    print("Images are saved inside the diffusion_steps folder.")
    print()

# 2. Simple Reverse Diffusion Using Gaussian Blur

if image is not None:
    noisy_image = noisy_images[-1]

    denoised_image = cv2.GaussianBlur(noisy_image, (5, 5), 0)

    cv2.imwrite("denoised_image.png", denoised_image)

    original = image
    comparison = np.hstack((original, noisy_image, denoised_image))

    cv2.imwrite("comparison.png", comparison)

    cv2.imshow("Original | Noisy | Denoised", comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("2. REVERSE DIFFUSION APPROXIMATION")
    print("-" * 40)
    print("Gaussian blur was used to reduce the noise.")
    print("The result is saved as denoised_image.png.")
    print("The visual comparison is saved as comparison.png.")
    print()

# 3. Stable Diffusion Image Generation

prompt = "A futuristic cricket stadium during IPL finals, night view, neon lights"

print("3. STABLE DIFFUSION")
print("-" * 40)
print("Prompt used:")
print(prompt)
print()
print("Generate the image using the Stable Diffusion web demo.")
print("Save the generated image as:")
print("ipl_stadium_generated.png")
print()

description = """
The AI interpreted the prompt as a modern cricket stadium during a major IPL final
match at night. It generated a futuristic atmosphere using bright neon lights,
stadium lighting, a large cricket ground, and a dramatic night-time environment.
"""

print("Description:")
print(description)

# 4. Real-World Applications of Diffusion Models

applications = {
    "Adobe Photoshop": "Diffusion-based generative features can be used for image inpainting, allowing users to remove or replace unwanted objects and fill the area with realistic content.",
    "Google Photos": "AI-based image restoration and enhancement can use generative techniques to improve low-quality images and reconstruct missing or unclear visual details."
}

print("4. REAL-WORLD APPLICATIONS")
print("-" * 40)

for application, description in applications.items():
    print(application + ":")
    print(description)
    print()

# 5. Risks and Ethical Concerns

risks = [
    {
        "risk": "Misinformation and Fake Images",
        "concern": "Text-to-image models can create realistic images of events or people that never actually happened. These images can be shared on social media and mislead users.",
        "solution": "Developers can add AI-generated content labels, invisible watermarks, and content provenance information to help users identify generated images."
    },
    {
        "risk": "Copyright and Unwanted Content",
        "concern": "Image generation models may create content that resembles copyrighted work or generate inappropriate content that can be misused on social media.",
        "solution": "Developers can use safety filters, content moderation systems, and better training-data policies to reduce harmful or unauthorized generations."
    }
]

print("5. RISKS AND ETHICAL CONCERNS")
print("-" * 40)

for item in risks:
    print("Risk:", item["risk"])
    print("Concern:", item["concern"])
    print("Possible Solution:", item["solution"])
    print()
