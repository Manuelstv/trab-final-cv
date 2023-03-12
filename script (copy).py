import os
path = "/home/msnuel/SphereNet-pytorch/images"
value = 1
for file in os.listdir(path):
    new_filename = f'{path}/wallpaper{value}.jpg'
    new_filename2 = f'{path}/wallpaper{value}.xml'
    if file.endswith('jpg'):
        if os.path.exists(new_filename):
            pass
        else:
            os.rename(file, new_filename)
            os.rename(f'{file[:-3]}xml', new_filename_2)
            value += 1

