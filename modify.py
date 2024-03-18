import os
import shutil

# Specify the directories to compare
dir1 = "D:\BE-Project-23-24\model\Vivo"  # Replace with the actual path to directory A
dir2 = "D:\BE-Project-23-24\model\Xiaomi"  # Replace with the actual path to directory B

dir1_list=os.listdir(dir1)
dir2_list=os.listdir(dir2)
dir1_list.sort()
dir2_list.sort()
print(dir1_list)
print(dir2_list)
for i in dir1_list:
    if not i.endswith(".csv"):
        dir1_list.remove(i)

for i in dir2_list:
    if not i.endswith(".csv"):
        dir2_list.remove(i)


for i in range(len(dir1_list)):
    file1_path = os.path.join(dir1, dir1_list[i])
    file2_path = os.path.join(dir2, dir2_list[i])
    new_file2_path = os.path.join(dir2, "x"+dir1_list[i][1:])  # Construct new path with filename from dir1
    if os.path.exists(new_file2_path):
        print("already_there")
        continue
    try:
        os.rename(file2_path, new_file2_path)
        print(f"Renamed file: {file2_path} -> {new_file2_path}")
    except OSError as error:
        print(f"Error renaming file: {error}")