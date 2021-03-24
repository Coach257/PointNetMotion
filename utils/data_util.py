import os
import json
import numpy as np

"""
Get all data files
"""
def get_all_files(data_path, ext = ".json"):
    data_files = []
    if(os.path.exists(os.path.join(data_path,"data_files.meta"))):
        with open(os.path.join(data_path,"data_files.meta")) as f:
            for line in f:
                data_files.append(line.strip())
        return data_files
    
    for file_dir, _ , files in os.walk(data_path):
        for f in files:
            if ext not in f:
                continue
            data_files.append(os.path.join(file_dir,f).replace(data_path,""))
    
    with open(os.path.join(data_path,"data_files.meta")) as f:
        for data_file in data_files:
            f.write(data_file + "\n")
    
    return data_files