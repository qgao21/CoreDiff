import os
import argparse
import numpy as np
import pydicom
from natsort import natsorted
from glob import glob


def save_dataset(args):
    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)
        print('Create path : {}'.format(args.save_root))

    for folder_ind in range(33):
        file_list = natsorted(glob(os.path.join(args.data_root, "SE{}".format(folder_ind), "IM*")))
        data = pydicom.dcmread(file_list[0])
        recon_type = data.SeriesDescription
        if "FBP" in recon_type:
            print(recon_type)
            print(len(file_list))
            if "FULL" in recon_type:
                dose = 'target'
            elif "50" in recon_type:
                dose = '50'
            elif "25" in recon_type:
                dose = '25'
            elif "10" in recon_type:
                dose = '10'
            elif "5" in recon_type:
                dose = '5'

            for file_ind in range(21, 871):
                data = pydicom.dcmread(file_list[file_ind])
                img = np.array(data.pixel_array)
                img[img < 0] = 0

                f = img
                f_name = 'piglet_' + dose + '_{:0>3d}_img.npy'.format(file_ind)
                np.save(os.path.join(args.save_root, f_name), f.astype(np.uint16))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/your data save path/piglet/DICOM/PA0/ST0/')   # data format: dicom
    parser.add_argument('--save_root', type=str, default='./gen_data/piglet_npy/')
    args = parser.parse_args()

    save_dataset(args)