import os
import random
import shutil

fileDir = 'graph'
trainDir = 'training'


def moveFile(fileDir, trainDir):
    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    filenumber = len(pathDir)
    # rate1 = 0.8  # 自定义抽取csv文件的比例，比方说100张抽80个，那就是0.8
    rate1 = 0.8
    picknumber1 = int(filenumber * rate1)  # 按照rate比例从文件夹中取一定数量的文件
    sample1 = random.sample(pathDir, picknumber1)  # 随机选取picknumber数量的样本
    for name in sample1:
        shutil.move(fileDir + '\\' + name, trainDir + "\\" + name)


if __name__ == '__main__':
    moveFile(fileDir, trainDir)
    # file = os.listdir(fileDir)
    # for i in file:
    #     moveFile(fileDir + '\\' + i, trainDir)