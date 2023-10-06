import os.path as osp
from torch.nn import functional as F
import torch
import torchvision
import argparse
import tqdm
import copy
from utils.measure import *
from utils.loss_function import PerceptualLoss
from utils.ema import EMA

from models.basic_template import TrainTask
from .corediff_wrapper import Network, WeightNet
from .diffusion_modules import Diffusion

import wandb
from ipdb import set_trace


class corediff(TrainTask):
    @staticmethod
    def build_options():
        parser = argparse.ArgumentParser('Private arguments for training of different methods')
        parser.add_argument("--in_channels", default=3, type=int)
        parser.add_argument("--out_channels", default=1, type=int)
        parser.add_argument("--init_lr", default=2e-4, type=float)

        parser.add_argument('--update_ema_iter', default=10, type=int)
        parser.add_argument('--start_ema_iter', default=2000, type=int)
        parser.add_argument('--ema_decay', default=0.995, type=float)

        parser.add_argument('--T', default=10, type=int)

        parser.add_argument('--sampling_routine', default='ddim', type=str)
        parser.add_argument('--only_adjust_two_step', action='store_true')
        parser.add_argument('--start_adjust_iter', default=1, type=int)
        
        return parser

    def set_model(self):
        opt = self.opt
        self.ema = EMA(opt.ema_decay)
        self.update_ema_iter = opt.update_ema_iter
        self.start_ema_iter = opt.start_ema_iter
        self.dose = opt.dose
        self.T = opt.T
        self.sampling_routine = opt.sampling_routine
        self.context = opt.context
        
        denoise_fn = Network(in_channels=opt.in_channels)

        model = Diffusion(
            denoise_fn=denoise_fn,
            image_size=512,
            timesteps=opt.T,
            context=opt.context
        ).cuda()
    
        optimizer = torch.optim.Adam(model.parameters(), opt.init_lr)
        ema_model = copy.deepcopy(model)

        self.logger.modules = [model, ema_model, optimizer]
        self.model = model
        self.optimizer = optimizer
        self.ema_model = ema_model

        self.lossfn = nn.MSELoss()
        self.lossfn_sub1 = nn.MSELoss()

        self.reset_parameters()
    

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())


    def step_ema(self, n_iter):
        if n_iter < self.start_ema_iter:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)


    def train(self, inputs, n_iter):
        opt = self.opt
        self.model.train()
        self.ema_model.train()
        low_dose, full_dose = inputs
        low_dose, full_dose = low_dose.cuda(), full_dose.cuda()

        ## training process of CoreDiff
        gen_full_dose, x_mix, gen_full_dose_sub1, x_mix_sub1 = self.model(
            low_dose, full_dose, n_iter,
            only_adjust_two_step=opt.only_adjust_two_step,
            start_adjust_iter=opt.start_adjust_iter
        )

        loss = 0.5 * self.lossfn(gen_full_dose, full_dose) + 0.5 * self.lossfn_sub1(gen_full_dose_sub1, full_dose)
        loss.backward()

        if opt.wandb:
            if n_iter == opt.resume_iter + 1:
                wandb.init(project="your wandb project name")

        self.optimizer.step()
        self.optimizer.zero_grad()

        lr = self.optimizer.param_groups[0]['lr']
        loss = loss.item()
        self.logger.msg([loss, lr], n_iter)

        if opt.wandb:
            wandb.log({'epoch': n_iter, 'loss': loss})

        if n_iter % self.update_ema_iter == 0:
            self.step_ema(n_iter)


    @torch.no_grad()
    def test(self, n_iter):
        opt = self.opt
        self.ema_model.eval()

        psnr, ssim, rmse = 0., 0., 0.
        for low_dose, full_dose in tqdm.tqdm(self.test_loader, desc='test'):
            low_dose, full_dose = low_dose.cuda(), full_dose.cuda()

            gen_full_dose, direct_recons, imstep_imgs = self.ema_model.sample(
                batch_size = low_dose.shape[0],
                img = low_dose,
                t = self.T,
                sampling_routine = self.sampling_routine,
                n_iter=n_iter,
                start_adjust_iter=opt.start_adjust_iter,
            )

            full_dose = self.transfer_calculate_window(full_dose)
            gen_full_dose = self.transfer_calculate_window(gen_full_dose)
        
            data_range = full_dose.max() - full_dose.min()
            psnr_score, ssim_score, rmse_score = compute_measure(full_dose, gen_full_dose, data_range)
            psnr += psnr_score / len(self.test_loader)
            ssim += ssim_score / len(self.test_loader)
            rmse += rmse_score / len(self.test_loader)

        self.logger.msg([psnr, ssim, rmse], n_iter)

        if opt.wandb:
            wandb.log({'epoch': n_iter, 'PSNR': psnr, 'SSIM': ssim, 'RMSE': rmse})


    @torch.no_grad()
    def generate_images(self, n_iter):
        opt = self.opt
        self.ema_model.eval()
        low_dose, full_dose = self.test_images

        gen_full_dose, direct_recons, imstep_imgs = self.ema_model.sample(
                batch_size = low_dose.shape[0],
                img = low_dose,
                t = self.T,
                sampling_routine = self.sampling_routine,
                n_iter=n_iter,
                start_adjust_iter=opt.start_adjust_iter,
            )

        if self.context:
            low_dose = low_dose[:, 1].unsqueeze(1)

        b, c, w, h = low_dose.size()
        fake_imgs = torch.stack([low_dose, full_dose, gen_full_dose])
        fake_imgs = self.transfer_display_window(fake_imgs)
        fake_imgs = fake_imgs.transpose(1, 0).reshape((-1, c, w, h))
        self.logger.save_image(torchvision.utils.make_grid(fake_imgs, nrow=3),
                               n_iter, 'test_{}_{}'.format(self.dose, self.sampling_routine) + '_' + opt.test_dataset)


    def train_osl_framework(self, test_iter):
        opt = self.opt
        self.ema_model.eval()

        ''' Initialize WeightNet '''
        weightnet = WeightNet(weight_num=10).cuda()
        optimizer_w = torch.optim.Adam(weightnet.parameters(), opt.init_lr*10)
        lossfn = PerceptualLoss()

        ''' get imstep images of diffusion '''
        for i in range(len(self.test_dataset)-2):
            if i == opt.index:
                if opt.unpair:
                    low_dose, _ = self.test_dataset[i]
                    _, full_dose = self.test_dataset[i+2]
                else:
                    low_dose, full_dose = self.test_dataset[i]
        low_dose, full_dose = torch.from_numpy(low_dose).unsqueeze(0).cuda(), torch.from_numpy(full_dose).unsqueeze(0).cuda()

        gen_full_dose, direct_recons, imstep_imgs = self.ema_model.sample(
            batch_size=low_dose.shape[0],
            img=low_dose,
            t=self.T,
            sampling_routine=self.sampling_routine,
            start_adjust_iter=opt.start_adjust_iter,
        )

        inputs = imstep_imgs.transpose(0, 2).squeeze(0)
        targets = full_dose

        ''' train WeightNet '''
        input_patches, target_patches = self.get_patch(inputs, targets, patch_size=opt.patch_size, stride=32)
        input_patches, target_patches = input_patches.detach(), target_patches.detach()

        for n_iter in tqdm.trange(1, opt.osl_max_iter):
            weightnet.train()
            batch_ids = torch.from_numpy(np.random.randint(0, input_patches.shape[0], opt.osl_batch_size)).cuda()
            input = input_patches.index_select(dim = 0, index = batch_ids).detach()
            target = target_patches.index_select(dim = 0, index = batch_ids).detach()

            out, weights = weightnet(input)
            loss = lossfn(out, target)
            loss.backward()

            optimizer_w.step()
            optimizer_w.zero_grad()
            lr = optimizer_w.param_groups[0]['lr']
            self.logger.msg([loss, lr], n_iter)
            if opt.wandb:
                wandb.log({'epoch': n_iter, 'loss': loss})
        opt_image = weights * inputs
        opt_image = opt_image.sum(dim=1, keepdim=True)
        print(weights)

        ''' Calculate the quantitative metrics before and after weighting'''
        full_dose_cal = self.transfer_calculate_window(full_dose)
        gen_full_dose_cal = self.transfer_calculate_window(gen_full_dose)
        opt_image_cal = self.transfer_calculate_window(opt_image)
        data_range = full_dose_cal.max() - full_dose_cal.min()
        psnr_ori, ssim_ori, rmse_ori = compute_measure(full_dose_cal, gen_full_dose_cal, data_range)
        psnr_opt, ssim_opt, rmse_opt = compute_measure(full_dose_cal, opt_image_cal, data_range)
        self.logger.msg([psnr_ori, ssim_ori, rmse_ori], test_iter)
        self.logger.msg([psnr_opt, ssim_opt, rmse_opt], test_iter)

        fake_imgs = torch.cat((low_dose[:, 1].unsqueeze(1), full_dose, gen_full_dose, opt_image), dim=0)
        fake_imgs = self.transfer_display_window(fake_imgs)
        self.logger.save_image(torchvision.utils.make_grid(fake_imgs, nrow=4), test_iter,
                               'test_opt_' + opt.test_dataset + '_{}_{}'.format(self.dose, opt.index))

        if opt.unpair:
            filename = './weights/unpair_weights_' + opt.test_dataset + '_{}_{}.npy'.format(self.dose, opt.index)
        else:
            filename = './weights/weights_' + opt.test_dataset + '_{}_{}.npy'.format(self.dose, opt.index)
        np.save(filename, weights.detach().cpu().squeeze().numpy())


    def test_osl_framework(self, test_iter):
        opt = self.opt
        self.ema_model.eval()

        if opt.unpair:
            filename = './weights/unpair_weights_' + opt.test_dataset + '_{}_{}.npy'.format(self.dose, opt.index)
        else:
            filename = './weights/weights_' + opt.test_dataset + '_{}_{}.npy'.format(self.dose, opt.index)
        weights = np.load(filename)
        print(weights)
        weights = torch.from_numpy(weights).unsqueeze(1).unsqueeze(2).unsqueeze(0).cuda()

        psnr_ori, ssim_ori, rmse_ori = 0., 0., 0.
        psnr_opt, ssim_opt, rmse_opt = 0., 0., 0.

        for low_dose, full_dose in tqdm.tqdm(self.test_loader, desc='test'):
            low_dose, full_dose = low_dose.cuda(), full_dose.cuda()

            gen_full_dose, direct_recons, imstep_imgs = self.ema_model.sample(
                batch_size=low_dose.shape[0],
                img=low_dose,
                t=self.T,
                sampling_routine=self.sampling_routine,
                n_iter=test_iter,
                start_adjust_iter=opt.start_adjust_iter,
            )
            imstep_imgs = imstep_imgs[:self.T]
            inputs = imstep_imgs.squeeze(2).transpose(0, 1)

            opt_image = weights * inputs
            opt_image = opt_image.sum(dim=1, keepdim=True)

            full_dose = self.transfer_calculate_window(full_dose)
            gen_full_dose = self.transfer_calculate_window(gen_full_dose)
            opt_image = self.transfer_calculate_window(opt_image)

            data_range = full_dose.max() - full_dose.min()
            psnr_ori, ssim_ori, rmse_ori = compute_measure(full_dose, gen_full_dose, data_range)
            psnr_opt, ssim_opt, rmse_opt = compute_measure(full_dose, opt_image, data_range)

            psnr_ori += psnr_ori / len(self.test_loader)
            ssim_ori += ssim_ori / len(self.test_loader)
            rmse_ori += rmse_ori / len(self.test_loader)

            psnr_opt += psnr_opt / len(self.test_loader)
            ssim_opt += ssim_opt / len(self.test_loader)
            rmse_opt += rmse_opt / len(self.test_loader)

        self.logger.msg([psnr_ori, ssim_ori, rmse_ori], test_iter)
        self.logger.msg([psnr_opt, ssim_opt, rmse_opt], test_iter)



    def get_patch(self, input_img, target_img, patch_size=256, stride=32):
        input_patches = []
        target_patches = []
        _, c_input, h, w = input_img.shape
        _, c_target, h, w = target_img.shape

        Top = np.arange(0, h - patch_size + 1, stride)
        Left = np.arange(0, w - patch_size + 1, stride)
        for t_idx in range(len(Top)):
            top = Top[t_idx]
            for l_idx in range(len(Left)):
                left = Left[l_idx]
                input_patch = input_img[:, :, top:top + patch_size, left:left + patch_size]
                target_patch = target_img[:, :, top:top + patch_size, left:left + patch_size]
                input_patches.append(input_patch)
                target_patches.append(target_patch)

        input_patches = torch.stack(input_patches).transpose(0, 1).reshape((-1, c_input, patch_size, patch_size))
        target_patches = torch.stack(target_patches).transpose(0, 1).reshape((-1, c_target, patch_size, patch_size))
        return input_patches, target_patches