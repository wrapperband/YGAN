"""
@author: Sean A. Cantrell and Robin N. Tully
"""
import codecs
import csv
from GAN import GAN



if __name__ == '__main__':
    with codecs.open('./data/image_coco.txt','r',encoding='utf-8',errors='replace') as file:
        data = [row[0] for row in csv.reader(file)]
    gan = GAN(128, 128, 100)
    gan.train(data, 128, 10, phase=3)
