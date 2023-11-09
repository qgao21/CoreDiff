import os
import os.path as osp
import argparse
import numpy as np
from natsort import natsorted
from glob import glob
import pydicom
import sys


def save_dataset(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print('Create path : {}'.format(args.save_path))

    patient_ids = ['C052', 'C232', 'C016', 'C120', 'C050', 'L077', 'L056', 'L186', 'L006', 'L148']

    for idx, patient_id in enumerate(patient_ids):
        print(idx)
        patient_path = osp.join(args.dataset_path, patient_id)
        for dir_path, dir_names, file_names in os.walk(patient_path):
            break
        patient_path = osp.join(patient_path, dir_names[0])
        for dir_path, dir_names, file_names in os.walk(patient_path):
            break

        for _, dir_name in enumerate(dir_names):
            if dir_name.find("Full") != -1 or dir_name.find("FULL") != -1:
                dose = 'target'
            elif dir_name.find("Low") != -1 or dir_name.find("LOW") != -1:
                if patient_id.startswith('C'):
                    dose = '10'
                elif patient_id.startswith('L'):
                    dose = '25'
            else:
                sys.stderr.write('Error dir_name!\n')
                raise SystemExit(1)

            data_paths = natsorted(glob(osp.join(patient_path, dir_name, '*.dcm')))
            for slice, data_path in enumerate(data_paths):
                im = pydicom.dcmread(data_path)
                f = np.array(im.pixel_array)

                f_name = '{}_'.format(patient_id) + dose + '_{:0>3d}_img.npy'.format(slice)
                np.save(os.path.join(args.save_path, f_name), f.astype(np.float32))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='/your data save path/')   # data format: dicom
    parser.add_argument('--save_path', type=str, default='./gen_data/mayo_2020_npy/')

    args = parser.parse_args()
    save_dataset(args)
