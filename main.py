import os
import torch
from torch.utils.data import DataLoader
from dataloader import CustomDataset
import torch.optim as optim
from tqdm import tqdm
import net as mm
import utils

batch_size = 16

description = 'pepsi'

save_path = './mid_test/' + description
model_path = './model/' + description

restore = False
restore_point = 300000
Checkpoint = model_path + '/cVG iter ' + str(restore_point) + '/Train_' + str(restore_point) + '.pth'

if not restore:
    restore_point = 0

saving_iter = 20000
Max_iter = 500000

# --------- Train

def main():

    # --------- Data
    print('Data Load')

    dPath_train = 'D:/dataset/RORD/train/'
    dPath_test = 'D:/dataset/RORD/test/'

    custom_train = CustomDataset(dPath_train)
    custom_test = CustomDataset(dPath_test, train=False)

    train_loader = DataLoader(custom_train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(custom_test, batch_size=1, shuffle=True)

    # --------- Module

    print('Model build')
    gen = mm.Generator().cuda()
    dis = mm.Discriminator().cuda()

    optimizer_G = optim.Adam(gen.parameters(), lr=0.0001, betas=(0, 0.9))
    optimizer_D = optim.Adam(dis.parameters(), lr=0.0004, betas=(0, 0.9))

    if restore:
        print('Weight Restoring.....')
        gen.load_state_dict(torch.load(Checkpoint)['G'])
        dis.load_state_dict(torch.load(Checkpoint)['D'])
        optimizer_G.load_state_dict(torch.load(Checkpoint)['opt_G'])
        optimizer_D.load_state_dict(torch.load(Checkpoint)['opt_D'])
        torch.cuda.empty_cache()
        print('Weight Restoring Finish!')

    print('Training start')
    is_training = True
    iter_count = restore_point
    for e in range(restore_point//train_loader.__len__(), 10000000):
        if not is_training:
            break
        for img, gt, msk in tqdm(train_loader, desc='Epoch\t%d\t' % e):
            Height_test, Width_test = img.shape[2:]
            if Height_test % 8 != 0 or Width_test % 8 != 0:
                ht = Height_test // 8 * 8
                wt = Width_test // 8 * 8
                img = img[:, :, :ht, :wt]
                gt = gt[:, :, :ht, :wt]
                msk = msk[:, :, :ht, :wt]
            if iter_count == int(Max_iter * 0.8):
                optimizer_G.param_groups[0]['lr'] = 0.00001
                optimizer_D.param_groups[0]['lr'] = 0.00004

            img_m = img.cuda() * msk.cuda()
            img_in, img_co = gen(img_m, msk.cuda())

            img_complete = img_in*(1-msk.cuda())+img.cuda()*msk.cuda()

            real_fake_image = torch.cat([gt.cuda(), img_complete.detach()], dim=0)

            dis_real_fake = dis(real_fake_image, torch.cat([msk, msk], dim=0).cuda())
            dis_real, dis_fake = torch.split(dis_real_fake, batch_size, dim=0)

            D_loss = torch.mean(torch.relu(1. - dis_real)) + torch.mean(torch.relu(1. + dis_fake))
            optimizer_D.zero_grad()
            D_loss.backward()
            optimizer_D.step()

            Loss_in = torch.mean(torch.abs(img_in - gt.cuda()))
            Loss_co = torch.mean(torch.abs(img_co - gt.cuda()))

            alpha = iter_count / Max_iter

            Loss_rec = Loss_in + 0.5*(1-alpha)*Loss_co

            dis_fake = dis(img_complete, msk.cuda())
            Loss_gan = -torch.mean(dis_fake)

            G_loss = 0.1*Loss_gan + 10*Loss_rec

            optimizer_G.zero_grad()
            G_loss.backward()
            optimizer_G.step()
            iter_count += 1

            if iter_count % 100 == 0:
                print('%d\t\tLoss = %.3f\tLoss_G = %.3f\tLoss_D = %.3f\t\tLr = %.5f'
                      % (iter_count, Loss_in.item(), Loss_gan.item(), D_loss.item(), optimizer_G.param_groups[0]['lr']))

            if iter_count % saving_iter == 0:

                print('SAVING MODEL...')
                utils.save_model(gen, dis, optimizer_G, optimizer_D, model_path, iter_count)
                print('SAVING MODEL Finish')

                print('VALIDATION...')
                log_name = save_path + '/eval.txt'
                psnr = utils.eval(test_loader, gen, save_path)
                torch.cuda.empty_cache()
                data = '%05d\tPSNR\t:\t%.2f\n' % (iter_count, psnr)
                print(data)

                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                with open(log_name, 'a+') as f:
                    f.write(data)

            if iter_count == Max_iter:
                is_training = False
                break

if __name__ == '__main__':
    main()