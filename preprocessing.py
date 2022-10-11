import numpy as np
import glob
import cv2
import os
from tqdm import tqdm
import json

d_Path = 'D:/dataset/inpainting/'
width = 480
height = 270

image_set_list = glob.glob(d_Path + '*/jpg/*')
np.random.shuffle(image_set_list)

def main():

    for i, di in enumerate(tqdm(image_set_list)):

        if i % 5 == 4:
            save_path = './test'
        else:
            save_path = './train'

        image_list = glob.glob(di + '/*.jpg')
        image_list.sort()
        path_gt = image_list[-1]

        image_gt = cv2.imread(path_gt)
        image_list = image_list[:-1]

        for path_image in tqdm(image_list, mininterval=10, desc='%s\t' % di):
            path_mask = path_image.replace('\\jpg', '\\M').replace('.jpg', '_M.json')
            name_image = path_image.split('\\')[-1].split('.')[0]

            Loadimage = cv2.imread(path_image)
            h, w, c = Loadimage.shape

            # load json file
            with open(path_mask, encoding='UTF8') as data_file:
                dic = json.load(data_file)['Learning_Data_Info.']['Annotation']

            Loadmask = np.zeros((h, w), np.uint8)

            Loadgt = Loadimage.copy()

            for dic_ann in dic:

                polys = dic_ann['segmentation']
                if len(polys) < 6:
                    continue
                polys = np.array(polys).reshape(-1, 2)
                mask_temp = cv2.fillPoly(np.zeros((h, w), np.uint8), [polys], 1)

                if mask_temp.sum() <= 1:
                    continue

                (x, y, bw, bh) = cv2.boundingRect(polys)

                cX = x + bw // 2
                cY = y + bh // 2
                Loadmask = mask_temp + Loadmask * (1 - mask_temp)

                yuv_target = cv2.cvtColor(image_gt, cv2.COLOR_BGR2YUV)
                y_target = yuv_target[:, :, 0]

                temp = cv2.copyMakeBorder(y_target, 10, 10, 10, 10, cv2.BORDER_REFLECT)
                temp[10:-10, 10:-10] = cv2.cvtColor(Loadgt, cv2.COLOR_BGR2YUV)[:, :, 0]
                mask_temp = cv2.copyMakeBorder(mask_temp, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
                dst = cv2.inpaint(temp, mask_temp, 3, cv2.INPAINT_NS)
                dst = dst[10:-10, 10:-10]
                mask_temp = mask_temp[10:-10, 10:-10]

                y_target = np.expand_dims(y_target, 2)
                dst = np.expand_dims(dst, 2)
                dst = np.tile(dst, (1, 1, 3))
                y_target = np.tile(y_target, (1, 1, 3))
                y_dst = cv2.seamlessClone(y_target, dst, mask_temp * 255, (cX, cY), cv2.NORMAL_CLONE)
                yuv_dst = np.concatenate((y_dst[:, :, 0:1], yuv_target[:, :, 1:3]), 2)
                Loadgt = cv2.cvtColor(yuv_dst, cv2.COLOR_YUV2BGR)

            Loadmask = 255 - Loadmask * 255
            Loadgt = cv2.resize(Loadgt, dsize=(width, height), interpolation=cv2.INTER_AREA)
            Loadimage = cv2.resize(Loadimage, dsize=(width, height), interpolation=cv2.INTER_AREA)
            Loadmask = cv2.resize(Loadmask, dsize=(width, height), interpolation=cv2.INTER_NEAREST)

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            save_name = save_path + '/img'
            if not os.path.exists(save_name):
                os.makedirs(save_name)

            cv2.imwrite(save_name + '/%s.png' % name_image, Loadimage)

            save_name = save_path + '/gt'
            if not os.path.exists(save_name):
                os.makedirs(save_name)

            cv2.imwrite(save_name + '/%s.png' % name_image, Loadgt)

            save_name = save_path + '/mask'
            if not os.path.exists(save_name):
                os.makedirs(save_name)

            cv2.imwrite(save_name + '/%s.png' % name_image, Loadmask)

if __name__ == '__main__':
    main()