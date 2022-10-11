from torch.utils.data import Dataset
import torchvision.transforms as trans
import cv2
import glob

class CustomDataset(Dataset):
    def __init__(self, img_root, train=True, height=256, width=256):

        self.train = train

        self.height = height
        self.width = width

        self.image_list = glob.glob(img_root + '/img/*')
        self.img_transform = trans.Compose([
            trans.ToTensor(),
            trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

        self.msk_transform = trans.ToTensor()

    def __getitem__(self, idx):
        path_image = self.image_list[idx]
        path_gt = path_image.replace('img', 'gt')
        path_mask = path_image.replace('img', 'mask').replace('jpg', 'png')

        Loadimage = cv2.imread(path_image)[:, :, ::-1]

        Loadgt = cv2.imread(path_gt)[:, :, ::-1]
        Loadmask = (cv2.imread(path_mask) != 0)
        Loadmask = Loadmask[:, :, 0:1].astype('float64')

        img = self.img_transform(Loadimage.copy())
        gt = self.img_transform(Loadgt.copy())
        msk = self.msk_transform(Loadmask.copy()).float()

        return img, gt, msk

    def __len__(self):
        return len(self.image_list)
