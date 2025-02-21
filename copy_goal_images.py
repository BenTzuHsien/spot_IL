import os, shutil

path = '/home/ben//RPM_Lab/Dataset_4Images/map01_01'
save_path = '/home/ben//RPM_Lab/Overlay Goal Images'

for p in sorted(os.listdir(path), key=lambda x: int(x[5:])):
    traj_path = os.path.join(path, p)
    step = sorted(os.listdir(traj_path))[-2]
    image_path = os.path.join(traj_path, step, '0.jpg')
    image_save_path = os.path.join(save_path, f'{p}.jpg')
    # print(image_path, image_save_path)
    shutil.copy(image_path, image_save_path)
