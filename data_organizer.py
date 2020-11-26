'''

This script separates mp4 files and npy files from each other if they're mixed
in the same folder

'''


import shutil, os

print("current working directory: ", os.getcwd())

original = "D:\\projects\\MadisonBB_video\\data\\basketballdataset\\examples"
mp4_path = "D:\\projects\\MadisonBB_video\\data\\basketballdataset\\mp4_files"
npy_path = "D:\\projects\\MadisonBB_video\\data\\basketballdataset\\npy_files"

if os.path.exists(original) and os.path.exists(mp4_path) and os.path.exists(npy_path):
    print("The paths exists!")

    for filename in os.listdir(original):

        if filename.endswith(".mp4"):
            shutil.move(original+"\\"+filename, mp4_path)

        if filename.endswith(".npy"):
             shutil.move(original+"\\"+filename, npy_path)

    print("Done moving files!")
else:
    print("Paths don't exist")


