# augmentation
import cv2

def augment_image(img):
    if img is None:
        return None
    
    # don't flip....unrealistic
    #img = cv2.flip(img, 1)
    
    # RESIZING of second image - should we do thisss??
    # img = cv2.resize(img, (256,256))
    
    # Adjust contrast (alpha) and brightness (beta)
    img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
    # Apply blur to test descriptor stability
    img = cv2.GaussianBlur(img, (3, 3), 0)
    #rotating
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    angle = 0 # Change this to your desired degree
    scale = 1.0

    # 1. Get the rotation matrix
    # getRotationMatrix2D(center, angle, scale)
    M = cv2.getRotationMatrix2D(center, angle, scale)

    # 2. Rotate the image
    # warpAffine(src, matrix, output_size)
    img = cv2.warpAffine(img, M, (w, h))    
    return img