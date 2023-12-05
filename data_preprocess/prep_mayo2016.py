import os
import os.path as osp
import argparse
import numpy as np
from natsort import natsorted
from glob import glob
import pydicom


def save_dataset(args):
    if not osp.exists(args.save_path):
        os.makedirs(args.save_path)
        print('Create path : {}'.format(args.save_path))

    patient_ids = ['L067', 'L096', 'L109', 'L143', 'L192', 'L286', 'L291', 'L310', 'L333', 'L506']

    io = 'target'
    for p_ind, patient_id in enumerate(patient_ids):
        print(patient_id)
        if p_ind >= 0:
            patient_path = osp.join(args.data_path, patient_id, 'full_1mm')
            data_paths = natsorted(glob(osp.join(patient_path, '*.IMA')))
            for slice, data_path in enumerate(data_paths):
                im = pydicom.dcmread(data_path)
                f = np.array(im.pixel_array)
                f_name = '{}_{}_{:0>3d}_img.npy'.format(patient_id, io, slice)
                np.save(osp.join(args.save_path, f_name), f.astype(np.uint16))


    io = '25'
    for p_ind, patient_id in enumerate(patient_ids):
        print(patient_id)
        if p_ind >= 0:
            patient_path = osp.join(args.data_path, patient_id, 'quarter_1mm')
            data_paths = natsorted(glob(osp.join(patient_path, '*.IMA')))
            for slice, data_path in enumerate(data_paths):
                im = pydicom.dcmread(data_path)
                f = np.array(im.pixel_array)
                f_name = '{}_{}_{:0>3d}_img.npy'.format(patient_id, io, slice)
                np.save(osp.join(args.save_path, f_name), f.astype(np.uint16))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/your data save path/')   # data format: dicom
    parser.add_argument('--save_path', type=str, default='./gen_data/mayo_2016_npy/')
    args = parser.parse_args()

    save_dataset(args)
