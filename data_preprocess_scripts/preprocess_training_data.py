import os
import re

def get_directories(wd):
    dirs = []
    for x in os.listdir(wd):
        if os.path.isdir(f"{wd}/{x}"):
            dirs.append(x)
    return dirs

def clean_row_numbers(file_path):
    pattern = re.compile(r'^\s*\d+\s*')
    
    # Read, clean, and overwrite
    with open(file_path, 'r+', encoding='utf-8') as f:
        lines = f.readlines()
        f.seek(0)
        f.truncate()
        for line in lines:
            cleaned_line = pattern.sub('', line)
            f.write(cleaned_line)


dirs = get_directories("../training_data")
for dir in dirs:
    for l_file in os.listdir(f"../training_data/{dir}"):
        if l_file.endswith(".txt"):
            clean_row_numbers(f"../training_data/{dir}/{l_file}")