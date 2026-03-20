import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct

# Load grayscale image
img = cv2.imread(r"c:\Users\Omprakash\Desktop\pthon\plain.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError("Image not found. Check the path!")

# Resize to multiple of 8
h, w = img.shape
h -= h % 8
w -= w % 8
img = img[:h, :w]

# Function for 2D DCT and inverse DCT
def block_dct(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def block_idct(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

# Apply block DCT
dct_blocks = np.zeros_like(img, dtype=float)
for i in range(0, h, 8):
    for j in range(0, w, 8):
        block = img[i:i+8, j:j+8]
        dct_block = block_dct(block)
        # Keep only top-left 4x4 coefficients (low frequencies)
        mask = np.zeros((8,8))
        mask[:4,:4] = 1
        dct_block = dct_block * mask
        dct_blocks[i:i+8, j:j+8] = dct_block

# Reconstruct image
recon = np.zeros_like(img, dtype=float)
for i in range(0, h, 8):
    for j in range(0, w, 8):
        block = dct_blocks[i:i+8, j:j+8]
        recon_block = block_idct(block)
        recon[i:i+8, j:j+8] = recon_block

# Show results
plt.figure(figsize=(12,6))
plt.subplot(1,3,1); plt.title("Original"); plt.imshow(img, cmap='gray')
plt.subplot(1,3,2); plt.title("DCT Coefficients (log)"); plt.imshow(np.log(abs(dct_blocks)+1), cmap='gray')
plt.subplot(1,3,3); plt.title("Reconstructed (low freq only)"); plt.imshow(np.clip(recon,0,255), cmap='gray')
plt.show()
