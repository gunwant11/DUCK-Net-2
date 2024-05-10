#  implement mixput augmentation

import numpy as np
import cv2

def mixup_data(image1, image2, mask1, mask2, alpha):
    lam = np.random.beta(alpha, alpha)
    mixed_image = cv2.addWeighted(image1, lam, image2, 1 - lam, 0)
    mixed_mask = cv2.addWeighted(mask1, lam, mask2, 1 - lam, 0)
    return mixed_image, mixed_mask


def cutmix(image1, mask1, image2, mask2, beta=1.0):
    # Generate random box coordinates
    img_h, img_w = image1.shape[:2]
    lam = np.random.beta(beta, beta)
    bbx1, bby1, bbx2, bby2 = np.random.rand(4)
    bbx1, bby1 = int(min(img_w, max(0, bbx1 * img_w))), int(min(img_h, max(0, bby1 * img_h)))
    bbx2, bby2 = int(min(img_w, max(0, bbx2 * img_w))), int(min(img_h, max(0, bby2 * img_h)))

    # Mix the images
    mixed_image = image1.copy()
    mixed_image[bby1:bby2, bbx1:bbx2] = image2[bby1:bby2, bbx1:bbx2]

    # Mix the masks
    mixed_mask = mask1.copy()
    mixed_mask[bby1:bby2, bbx1:bbx2] = mask2[bby1:bby2, bbx1:bbx2]

    return mixed_image, mixed_mask

# Cutout augmentation
def cutout(image, mask_size=(20, 20)):
    # Randomly choose position for the cutout mask
    h, w = image.shape[:2]
    mask_h, mask_w = mask_size
    top = np.random.randint(0, h - mask_h)
    left = np.random.randint(0, w - mask_w)

    # Apply the cutout mask
    mask = np.ones((h, w), np.uint8)
    mask[top:top + mask_h, left:left + mask_w] = 0
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    return masked_image
