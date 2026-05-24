import torch
import torch.nn.functional as F




def compute_local_mean(x, kernel_size=3):
    # Extract the input dimensions
    B, C, H, W = x.shape

    # Create a uniform square kernel of kernel_size
    uniform_kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2)

    # Expand the kernel to cover all channels in the image
    uniform_kernel = uniform_kernel.expand(C, 1, -1, -1).to(x.device)

    # Compute the grouped mean
    return F.conv2d(x, uniform_kernel, padding=kernel_size//2, groups=C)

def ssim(img1, img2, kernel_size=3, k1=0.01, k2=0.03):

    # Normalize both images to allow working with dynamic range of L=1
    img1 = img1 - torch.min(img1)
    img1 = img1 / torch.max(img1)

    img2 = img2 - torch.min(img2)
    img2 = img2 / torch.max(img2)

    # Compute c1 and c2 (mult. by L omitted as L=1)
    c1 = k1**2
    c2 = k2**2

    # Compute spatial mean (localized) for both images
    mu_x = compute_local_mean(img1, kernel_size).to(img1.device)
    mu_y = compute_local_mean(img2, kernel_size).to(img1.device)

    mu_x2 = compute_local_mean(img1 * img1, kernel_size).to(img1.device)
    mu_y2 = compute_local_mean(img2 * img2, kernel_size).to(img1.device)

    mu_xy = compute_local_mean(img1 * img2, kernel_size).to(img1.device)

    # Compute variances
    sigma_x2 = mu_x2 - mu_x*mu_x
    sigma_y2 = mu_y2 - mu_y*mu_y
    sigma_xy = mu_xy - mu_x*mu_y

    # Compute the spatial SSIM
    nominator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x*mu_x + mu_y*mu_y + c1) * (sigma_x2 + sigma_y2 + c2)

    return nominator / denominator