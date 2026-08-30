# Satellite Image Super-Resolution Using Diffusion Models

A deep learning project for enhancing the resolution of satellite imagery using a **Denoising Diffusion Probabilistic Model (DDPM)** with a U-Net-based noise prediction network.

The project generates paired low-resolution (LR) and high-resolution (HR) satellite images from the **EuroSAT dataset** and trains a diffusion model to reconstruct high-resolution image details from degraded inputs.

## Project Overview

Satellite imagery is widely used in applications such as land-use analysis, agriculture, environmental monitoring, and geographic information systems. However, low-resolution imagery can lose important spatial details.

This project explores **image super-resolution using diffusion-based generative modeling**. High-resolution satellite images are artificially downscaled to create low-resolution inputs. The diffusion model then learns to reconstruct the high-resolution representation while using the low-resolution image as conditioning information.

## Key Features

* Satellite image super-resolution using a diffusion model
* EuroSAT dataset for satellite imagery
* Automatic generation of LR-HR image pairs
* HSV-independent RGB image processing
* U-Net-based noise prediction network
* DDPM forward and reverse diffusion processes
* Conditional reconstruction using LR images
* GPU acceleration using PyTorch CUDA when available
* MSE-based noise prediction loss
* Visualization of HR, LR, and super-resolved outputs
* Model checkpoint saving using PyTorch

## Model Pipeline

```text
              EuroSAT Images
                    │
                    ▼
          High-Resolution Image
                    │
              Downsampling
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
     Low-Resolution       High-Resolution
        Image                 Image
          │                     │
          ▼                     ▼
      Conditioning       Forward Diffusion
                                │
                                ▼
                         Noisy HR Image
                                │
                    ┌───────────┴───────────┐
                    │                       │
              Noisy HR Image          LR Image
                    │                       │
                    └───────────┬───────────┘
                                ▼
                         U-Net Model
                                │
                                ▼
                       Predicted Noise
                                │
                                ▼
                       Reverse Diffusion
                                │
                                ▼
                    Super-Resolved Image
```

## Dataset

The project uses the **EuroSAT dataset**, which contains satellite images covering different land-use and land-cover categories.

The notebook loads the dataset using the Hugging Face `datasets` library and creates low-resolution/high-resolution pairs from the original images.

The high-resolution images are resized to **128 × 128 pixels**.

The low-resolution images are first reduced by a **4× scale factor** and then resized back to 128 × 128 pixels. This creates a degraded image that the model attempts to reconstruct.

## Data Preprocessing

Each image is converted to RGB and transformed into a PyTorch tensor.

The images are normalized from:

```text
[0, 1]
```

to:

```text
[-1, 1]
```

The preprocessing pipeline creates:

```text
HR Image → 128 × 128
LR Image → 32 × 32 → 128 × 128
```

This provides paired LR and HR images for conditional super-resolution training.

## Diffusion Model

The project implements a simplified **Denoising Diffusion Probabilistic Model (DDPM)**.

A diffusion schedule with:

```text
T = 500
```

timesteps is used, with beta values linearly increasing from:

```text
1e-4 → 0.02
```

### Forward Diffusion

During training, Gaussian noise is progressively added to the high-resolution image.

The forward process follows the formulation:

```text
xₜ = √α̅ₜ x₀ + √(1 − α̅ₜ) ε
```

where:

* `x₀` = original high-resolution image
* `xₜ` = noisy image at timestep `t`
* `α̅ₜ` = cumulative product of the diffusion coefficients
* `ε` = sampled Gaussian noise

The model is trained to predict the noise added to the high-resolution image.

## U-Net Architecture

A U-Net-inspired convolutional architecture is used as the noise prediction network.

The implementation contains:

* Initial convolution layer
* Multiple residual blocks
* Group normalization
* SiLU activation functions
* Final convolution layer

The model receives a combination of:

```text
Noisy HR Image + LR Image
```

as its input.

These tensors are concatenated along the channel dimension to provide the model with both the noisy image and low-resolution conditioning information.

## Residual Blocks

The network uses residual blocks consisting of:

* 3×3 convolution
* Group Normalization
* SiLU activation
* 3×3 convolution
* Group Normalization
* Residual connection

Residual connections help the network learn transformations while preserving useful information from earlier layers.

## Training

The model is trained using:

```text
Optimizer: Adam
Learning Rate: 1e-4
Epochs: 5
Batch Size: 16
Diffusion Timesteps: 500
Image Size: 128 × 128
```

The primary training objective is **mean squared error (MSE)** between the actual noise added during forward diffusion and the noise predicted by the U-Net.

```python
loss = F.mse_loss(noise_pred, noise)
```

The trained model parameters are saved as:

```text
eurosat_sr_ddpm.pth
```

## Inference

During inference, the model starts from a noisy image representation and performs reverse diffusion over the 500 diffusion steps.

At every step:

1. The current noisy image is combined with the LR image.
2. The U-Net predicts the noise.
3. The DDPM reverse process removes part of the predicted noise.
4. The process continues until the final reconstructed image is obtained.

The notebook visualizes:

```text
Original HR Image | Low-Resolution Image | Super-Resolved Image
```

for samples from the different EuroSAT categories.

## Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Hugging Face Datasets
* KaggleHub
* Matplotlib
* PIL
* Einops
* CUDA
* Deep Learning
* Generative AI
* Diffusion Models
* Computer Vision
* Image Super-Resolution

## Hardware / Compute

The model automatically detects CUDA availability and uses a GPU when available:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

The notebook is therefore suitable for execution in GPU-enabled environments such as Google Colab or Kaggle.

## Results

The notebook provides qualitative comparisons between:

* Original high-resolution satellite images
* Artificially downscaled low-resolution images
* Diffusion-model super-resolved outputs

The comparison is generated across the available EuroSAT categories to examine how the model reconstructs different types of satellite scenes.

## Limitations

This implementation is intended as an experimental implementation of diffusion-based super-resolution and has several limitations:

* The training configuration uses only 5 epochs.
* The U-Net architecture is relatively lightweight compared with modern diffusion architectures.
* The timestep embedding is not explicitly used in the implemented U-Net.
* Reconstruction quality is evaluated primarily through visual comparison.
* The current implementation does not report quantitative SR metrics such as PSNR or SSIM.
* The implementation described as using a VAE in the project overview does not currently include an implemented VAE encoder/decoder in the training pipeline.

## Future Improvements

* Add PSNR and SSIM evaluation
* Train for more epochs
* Implement timestep embeddings in the U-Net
* Develop a deeper multi-scale U-Net architecture
* Add perceptual loss
* Implement a proper latent diffusion architecture with VAE encoding/decoding
* Experiment with different diffusion schedules
* Compare against CNN-based super-resolution models
* Compare against established architectures such as SRCNN, SRGAN, or ESRGAN
* Perform quantitative evaluation across different EuroSAT categories

## How to Run

Clone the repository:

```bash
git clone https://github.com/shriyyaaa/satellite-image-super-resolution-diffusion.git
cd satellite-image-super-resolution-diffusion
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Open the notebook:

```text
[satellite-image-super-resolution.ipynb](https://colab.research.google.com/drive/18OGVyRiBaF9mjh8zq7RpKOCyJIhUcAVg?usp=sharing)
```

The project can also be executed in Google Colab using a GPU runtime.

## Author

**Shriya Tiwari**

Computer Science Engineering
Thapar Institute of Engineering and Technology
