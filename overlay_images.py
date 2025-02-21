import os, cv2
import numpy as np

path = '/home/ben//RPM_Lab/Overlay Goal Images'
total_images = np.zeros([720, 1080, 3])
total_num_images = 0

for image_name in os.listdir(path)[0:2]:
    image = cv2.imread(os.path.join(path, image_name))
    total_num_images += 1
    total_images += image

print(total_num_images)
total_images = total_images / total_num_images
total_images = np.clip(total_images, 0, 255).astype(np.uint8)
cv2.imshow('total', total_images)
cv2.waitKey(0)
    
