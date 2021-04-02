import json
import os
import random
import shutil
from utils.data_util import get_all_files
data_path = "/home/data/Motion3D/CatGirl_SMPL_json_Results/CatGirl_SMPL_json/CatGirl_SMPL_json/"
data_main_path = "/home/data/Motion3D/CatGirl_SMPL_json_Results/"


def get_file_info(file_path):
    file_num = file_path.split("_")[-1].replace(".json","")
    file_name = file_path.replace("_" + file_num+".json","")
    return file_name

def movefile(orgfile, desfile):
    desdir = desfile.replace(desfile.split("/")[-1],"")
    if not os.path.exists(desdir):
        os.makedirs(desdir)
    shutil.move(orgfile,desfile)

def main():
    data_files = get_all_files(data_path)
    while len(data_files) != 0 :
        file_name = get_file_info(data_files[0])
        data_list = [data_files[0]]
        for data_file in data_files:
            if file_name not in data_file:
                continue
            data_list.append(data_file)
            data_files.remove(data_file)
        random.shuffle(data_list)
        if len(data_list) > 1:
            print("Move {} to {}".format(os.path.join(data_path,data_list[0]),os.path.join(data_main_path,"CatGirl_SMPL_json/",data_list[0])))
            movefile(os.path.join(data_path,data_list[0]),os.path.join(data_main_path,"CatGirl_SMPL_json/",data_list[0]))

if __name__ == "__main__":
    main()