import os
import argparse
import numpy as np
import h5py
from ipdb import set_trace


def save_dataset(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print('Create path : {}'.format(args.save_path))

    patients_list = sorted([d for d in os.listdir(args.data_path) if '_full_1mm_CT.mat' in d])
    for p_ind, patient in enumerate(patients_list):
        print(patient)
        if p_ind >= 0:
            patient_id = patient.split('_', 1)[0]
            patient_path = os.path.join(args.data_path, patient)
            Img = h5py.File(patient_path)
            Img = Img['Img_CT'][:].transpose(0, 2, 1)   # (slices ,512, 512)

            io = 'target'
            for slice in range(Img.shape[0]):
                f = Img[slice]
                f_name = '{}_{}_{:0>3d}_img.npy'.format(patient_id, io, slice)
                np.save(os.path.join(args.save_path, f_name), f.astype(np.uint16))


    patients_list = sorted([d for d in os.listdir(args.data_path) if '_quarter_1mm_CT.mat' in d])
    for p_ind, patient in enumerate(patients_list):
        print(patient)
        if p_ind >= 0:
            patient_id = patient.split('_', 1)[0]
            patient_path = os.path.join(args.data_path, patient)
            Img = h5py.File(patient_path)
            Img = Img['Img_CT'][:].transpose(0, 2, 1)

            dose = 25
            for slice in range(Img.shape[0]):
                f = Img[slice]
                f_name = '{}_{}_{:0>3d}_img.npy'.format(patient_id, dose, slice)
                np.save(os.path.join(args.save_path, f_name), f.astype(np.uint16))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/your data save path/')   # data format: matlab
    parser.add_argument('--save_path', type=str, default='./gen_data/mayo_2016_sim_npy/')
    args = parser.parse_args()

    save_dataset(args)