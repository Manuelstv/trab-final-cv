import os
path = "/home/msnuel/SphereNet-pytorch/images"
value = 1
for file in os.listdir(path):
    new_filename = f'wallpaper{value}.jpg'
    new_filename2 = f'wallpaper{value}.xml'
    if new_filename.endswith('jpg'):
        while os.path.exists(new_filename):
            value += 1
            new_filename = f'wallpaper{value}.jpg'
            new_filename2 = f'wallpaper{value}.xml'
        os.rename(file, new_filename)
        os.rename(f'{file[:-3]}xml', new_filename2)
        value += 1

