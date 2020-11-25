import shutil, os

print("current working directory: ", os.getcwd())

if os.path.exists(str(os.getcwd) + "\\dataset\\examples"):
    print("the path exists!")
else:
    print("This path doesn't exist!")

