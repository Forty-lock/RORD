import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloader import CustomDataset
import torchvision
import net as mm
import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim

description = 'pepsi_place2'
save_path = './results/' + description

model_path = './model/' + description

restore_point = 500000
Checkpoint = model_path + '/cVG iter ' + str(restore_point) + '/Train_' + str(restore_point) + '.pth'

def main():
    # --------- Data
    print('Data Load')
    dPath_data = 'D:/dataset/inpainting/COCO/test/'
    custom_data = CustomDataset(dPath_data, train=False)
    data_loader = DataLoader(custom_data, batch_size=1)

    # --------- Model
    print('Model build')
    network = mm.Generator().cuda()

    print('Weight Restoring.....')
    network.load_state_dict(torch.load(Checkpoint)['G'])
    torch.cuda.empty_cache()
    print('Weight Restoring Finish!')

    # --------- Test
    temp1 = []
    temp2 = []
    network.eval()
    for isave, (img, gt, msk) in enumerate(tqdm(data_loader)):

        _, c_test, Height_test, Width_test = img.shape
        if Height_test < 128 or Width_test < 128:
            continue

        if Height_test * Width_test > 310000:
            continue

        if Height_test % 64 != 0 or Width_test % 64 != 0:
            ht = Height_test // 64 * 64
            wt = Width_test // 64 * 64
            img = img[:, :, :ht, :wt]
            gt = gt[:, :, :ht, :wt]
            msk = msk[:, :, :ht, :wt]
            _, c_test, Height_test, Width_test = img.shape

        rate = msk.mean()
        if rate > 0.3 and rate != 1:
            prediction = network(img.cuda() * msk.cuda(), msk.cuda()).cpu()

            Bigpaper = torch.cat([img, img * msk, gt, prediction], 0)

            img = (torch.permute(prediction[0], (1, 2, 0)) + 1) / 2 * 255
            img = img.detach().numpy()
            img = img.astype(np.uint8)

            gt = (torch.permute(gt[0], (1, 2, 0)) + 1) / 2 * 255
            gt = gt.detach().numpy()
            gt = gt.astype(np.uint8)

            grayA = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            grayB = cv2.cvtColor(gt, cv2.COLOR_BGR2YCrCb)

            psnr = cv2.PSNR(img, gt)
            score_y, diff = compare_ssim(grayA[:, :, 0], grayB[:, :, 0], full=True)
            score_cr, diff = compare_ssim(grayA[:, :, 1], grayB[:, :, 1], full=True)
            score_cb, diff = compare_ssim(grayA[:, :, 2], grayB[:, :, 2], full=True)

            score = 0.8*score_y + 0.1*score_cr + 0.1*score_cb

            temp1.append(psnr)
            temp2.append(score)

            if not os.path.exists(save_path + '/img'):
                os.makedirs(save_path + '/img')

            # if isave % 100 == 0:
            #     torchvision.utils.save_image(Bigpaper, save_path + '/img/img_%06d_%.2f.png' % (isave, psnr), normalize=True, nrow=2)

        else:
            print(isave)

    data = '%05d\tPSNR\t:\t%.2f\n' % (restore_point, np.mean(temp1))
    data2 = '%05d\tSSIM\t:\t%.3f\n' % (restore_point, np.mean(temp2))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(save_path + '/eval.txt', 'a+') as f:
        f.write(dPath_data)
        f.write('\n')

    with open(save_path + '/eval.txt', 'a+') as f:
        f.write(data)

    with open(save_path + '/eval.txt', 'a+') as f:
        f.write(data2)

if __name__ == '__main__':
    main()