import os
import os.path as osp
from glob import glob
from torch.utils.data import Dataset
import numpy as np
import torch
from functools import partial
import torch.nn.functional as F
from ipdb import set_trace


class CTDataset(Dataset):
    def __init__(self, dataset, mode, test_id=9, dose=5, context=True):
        self.mode = mode
        self.context = context

        if dataset == 'mayo_2016':
            npy_root = 'your Mayo-2016 data folder, data format: .npy'

            patient_ids = [67, 96, 109, 143, 192, 286, 291, 310, 333, 506]
            if mode == 'train':
                patient_ids.pop(test_id)
            elif mode == 'test':
                patient_ids = patient_ids[test_id:test_id + 1]

            patient_lists = []
            for ind, id in enumerate(patient_ids):
                patient_list = sorted(glob(osp.join(npy_root, ('L{:03d}_target_'.format(id) + '*_img.npy'))))
                patient_lists = patient_lists + patient_list[1:len(patient_list) - 1]
            base_target = patient_lists

            patient_lists = []
            for ind, id in enumerate(patient_ids):
                patient_list = sorted(glob(osp.join(data_root, ('L{:03d}_{}_'.format(id, dose) + '*_img.npy'))))
                if context:
                    cat_patient_list = []
                    for i in range(1, len(patient_list) - 1):
                        patient_path = ''
                        for j in range(-1, 2):
                            patient_path = patient_path + '~' + patient_list[i + j]
                        cat_patient_list.append(patient_path)
                    patient_lists = patient_lists + cat_patient_list
                else:
                    patient_list = patient_list[1:len(patient_list) - 1]
                    patient_lists = patient_lists + patient_list
            base_input = patient_lists

        elif dataset == 'mayo_2020':
            npy_root = 'your Mayo-2020 data folder, data format: .npy'
            patient_ids = ['C052', 'C232', 'C016', 'C120', 'C050', 'L077', 'L056', 'L186', 'L006', 'L148']

            patient_lists = []
            for ind, id in enumerate(patient_ids):
                patient_list = sorted(glob(osp.join(npy_root, (id + 'target' + '*_img.npy'))))
                patient_lists = patient_lists + patient_list[1:len(patient_list) - 1]
            base_target = patient_lists

            patient_lists = []
            for ind, id in enumerate(patient_ids):
                patient_list = sorted(glob(osp.join(npy_root, (id + 'input' + '*_img.npy'))))
                cat_patient_list = []
                for i in range(1, len(patient_list) - 1):
                    patient_path = ''
                    for j in range(-1, 2):
                        patient_path = patient_path + '~' + patient_list[i + j]
                    cat_patient_list.append(patient_path)
                print(len(cat_patient_list))
                patient_lists = patient_lists + cat_patient_list


        elif dataset == 'piglet':
            npy_root = 'your piglet data folder, data format: .npy'

            patient_list = sorted(glob(osp.join(npy_root, 'piglet_target_' + '*_img.npy')))
            base_target = patient_list[1:len(patient_list) - 1]

            patient_list = sorted(glob(osp.join(npy_root, 'piglet_{}_'.format(dose) + '*_img.npy')))
            if opt.context:
                cat_patient_list = []
                for i in range(1, len(patient_list) - 1):
                    patient_path = ''
                    for j in range(-1, 2):
                        patient_path = patient_path + '~' + patient_list[i + j]
                    cat_patient_list.append(patient_path)
                    base_input = cat_patient_list
            else:
                patient_list = patient_list[1:len(patient_list) - 1]
                patient_lists = patient_lists + patient_list


        elif dataset == 'phantom':
            npy_root = 'your phantom data folder, data format: .npy'

            patient_list = natsorted(glob(osp.join(npy_root, 'xnat_499' + '*_img.npy')))[9:21]
            base_target = patient_list[1:len(patient_list) - 1]

            patient_list = sorted(glob(osp.join(npy_root, 'xnat_{:0>3d}_'.format(dose) + '*_img.npy')))[9:21]
            cat_patient_list = []
            for i in range(1, len(patient_list) - 1):
                patient_path = ''
                for j in range(-1, 2):
                    patient_path = patient_path + '~' + patient_list[i + j]
                cat_patient_list.append(patient_path)
                base_input = cat_patient_list

        self.input = base_input
        self.target = base_target
        print(len(self.input))
        print(len(self.target))


    def __getitem__(self, index):
        input, target = self.input[index], self.target[index]

        if self.context:
            input = np.load(input).transpose([2, 1, 0]).astype(np.float32) #(3, 512, 512), 5 is the dose dimension
        else:
            input = np.load(input).transpose([1, 0])[np.newaxis, ...].astype(np.float32) #(1, 512, 512)
        target = np.load(target).transpose([1, 0])[np.newaxis,...].astype(np.float32) #(1, 512, 512)
        input = normalize_(input)
        target = normalize_(target)

        return input, target

    def __len__(self):
        return len(self.target)

    def normalize_(self, img, MIN_B=-1024, MAX_B=3072):
        img = img - 1024
        img[img < MIN_B] = MIN_B
        img[img > MAX_B] = MAX_B
        img = (img - MIN_B) / (MAX_B - MIN_B)
        return img


dataset_dict = {
    'train': partial(CTDataset, dataset='mayo_2016', mode='train', test_id=9, dose=5, context=True),
    'test': partial(CTDataset, dataset='mayo_2016', mode='test', test_id=9, dose=5, context=True),
}