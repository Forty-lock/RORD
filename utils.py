import torch
import torchvision
import cv2
import numpy as np
import os
import random
import math
from typing import List, Tuple

def ff_mask(height, width, maxLen=50, maxWid=25, maxNum=5, maxVer=25, minLen=10, minWid=5, minVer=5, Ang=3.14):

    mask = np.ones((height, width, 1))
    for n in range(random.randint(2, maxNum)):
        startX = random.randint(0, height)
        startY = random.randint(0, width)
        numVer = random.randint(minVer, maxVer)
        W = random.randint(minWid, maxWid)
        for j in range(numVer):
            angle = random.uniform(-Ang, Ang)
            length = random.randint(minLen, maxLen)

            endX = min(height-1, max(0, int(startX + length * math.sin(angle))))
            endY = min(width-1, max(0, int(startY + length * math.cos(angle))))

            cv2.circle(mask, (startX, startY), W, 0, -1)
            cv2.circle(mask, (endX, endY), W, 0, -1)
            cv2.line(mask, (startX, startY), (endX, endY), 0, 2*W)

            startX = endX
            startY = endY

    return mask

def poly_mask(height, width, mask=None):

    if mask is None:
        mask = np.ones((height, width, 1))
    for n in range(random.randint(1, 3)):

        if mask.mean() < 0.3:
            break

        startX = random.randint(0, height)
        startY = random.randint(0, width)
        numVer = random.randint(8, 15)
        R = random.randint(20, 80)

        poly = generate_polygon(center=(startY, startX),
                                avg_radius=R,
                                irregularity=0.5,
                                spikiness=0.3,
                                num_vertices=numVer)
        cv2.fillPoly(mask, [poly], 0)

    return mask

def generate_polygon(center: Tuple[float, float], avg_radius: float,
                     irregularity: float, spikiness: float,
                     num_vertices: int):
    # Parameter check
    if irregularity < 0 or irregularity > 1:
        raise ValueError("Irregularity must be between 0 and 1.")
    if spikiness < 0 or spikiness > 1:
        raise ValueError("Spikiness must be between 0 and 1.")

    irregularity *= 2 * math.pi / num_vertices
    spikiness *= avg_radius
    angle_steps = random_angle_steps(num_vertices, irregularity)

    # now generate the points
    points = []
    angle = random.uniform(0, 2 * math.pi)
    for i in range(num_vertices):
        radius = np.clip(random.gauss(avg_radius, spikiness), 0, 2 * avg_radius)
        point = (center[0] + radius * math.cos(angle),
                 center[1] + radius * math.sin(angle))
        points.append(point)
        angle += angle_steps[i]

    points = np.array(points, np.int32)

    return points

def random_angle_steps(steps: int, irregularity: float) -> List[float]:
    # generate n angle steps
    angles = []
    lower = (2 * math.pi / steps) - irregularity
    upper = (2 * math.pi / steps) + irregularity
    cumsum = 0
    for i in range(steps):
        angle = random.uniform(lower, upper)
        angles.append(angle)
        cumsum += angle

    # normalize the steps so that point 0 and point n+1 are the same
    cumsum /= (2 * math.pi)
    for i in range(steps):
        angles[i] /= cumsum
    return angles

def save_img(img, gt, msk, prediction, save_path, isave):

    save_path_img = save_path + '/img'
    if not os.path.exists(save_path_img):
        os.makedirs(save_path_img)

    Bigpaper = torch.cat([img, img*msk, gt, prediction], 0)

    name = save_path_img + '/img_%04d.png' % isave
    torchvision.utils.save_image(Bigpaper, name, normalize=True, nrow=2)

def eval(test_loader, model, save_path):
    model.eval()

    temp = []

    with torch.no_grad():
        for i, (img, gt, msk) in enumerate(test_loader):

            Height_test, Width_test = img.shape[2:]
            if Height_test % 64 != 0 or Width_test % 64 != 0:
                ht = Height_test // 64 * 64
                wt = Width_test // 64 * 64
                img = img[:, :, :ht, :wt]
                gt = gt[:, :, :ht, :wt]
                msk = msk[:, :, :ht, :wt]

            prediction = model(img.cuda()*msk.cuda(), msk.cuda()).cpu()

            if i < 30:
                save_img(img, gt, msk, prediction, save_path, i)

            gt = (gt + 1) * 255 / 2
            prediction = (prediction + 1) * 255 / 2
            mse = torch.mean((prediction - gt) ** 2).item()
            psnr = 20 * math.log10(255.0 / math.sqrt(mse + 1e-10))

            temp.append(psnr)

    model.train()
    return np.mean(temp)

def save_model(G, D, optimizer_g, optimizer_d, model_path, iter_count):

    Temp = model_path + '/cVG iter %s/' % iter_count

    if not os.path.exists(Temp):
        os.makedirs(Temp)

    SaveName = Temp + 'Train_%s.pth' % iter_count
    torch.save({
        'G': G.state_dict(),
        'D': D.state_dict(),
        'opt_G': optimizer_g.state_dict(),
        'opt_D': optimizer_d.state_dict(),
    }, SaveName)
    torch.cuda.empty_cache()
