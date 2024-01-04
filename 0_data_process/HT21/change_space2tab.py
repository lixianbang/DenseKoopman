import os
import re

os.getcwd()

names = ['train', 'val', 'test']
for split_name in names:
    path = f"./19spaces/19{split_name}/"  # 原始文件夹目录

    files = os.listdir(path)  # 得到文件夹下的所有文件名称

    for file in files:  # 遍历文件夹

        position = os.path.join(path, file)

        with open(position, "r", encoding='utf-8') as f:  # 打开文件
            for eachline in f.readlines():
                lines = eachline
                # 替换空格为tab
                lines = re.sub(' ', '\t', lines)

                # 创建新文件
                file_save = 'ntab_' + file
                outpath = os.path.join(path, file_save)
                out = open(outpath, 'a+')
                out.write(lines)
                f.close
                out.close
