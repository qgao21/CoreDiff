import os
import argparse
import numpy as np
import pydicom
from natsort import natsorted
from glob import glob
from skimage import transform


def save_dataset(args):
    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)
        print('Create path : {}'.format(args.save_root))

    for dir_path, dir_names, file_names in os.walk(args.data_root):
        if len(dir_names) and dir_path == args.data_root:
            dir_names = natsorted(dir_names)[1:]
            print(dir_names)
            dir_id = -1
            for _, dir_name in enumerate(dir_names):
                if 'THORAXXVI' not in dir_name:
                    file_list = natsorted(glob(os.path.join(args.data_root, dir_name, "resources/DICOM/files/", "*.dcm")))
                    data = pydicom.dcmread(file_list[0])
                    mAs = data.XRayTubeCurrent
                    if mAs == 499:
                        dir_id += 1
                        print(dir_id)
                    if dir_id == 4:
                        for file_ind in range(len(file_list)):
                            data = pydicom.dcmread(file_list[file_ind])
                            img = np.array(data.pixel_array).astype(np.float32)
                            img = transform.resize(img, (512, 512))
                            img[img<0] = 0

                            f = img
                            if mAs == 499:
                                f_name = 'xnat_target_{:0>3d}_img.npy'.format(file_ind)
                            else:
                                f_name = 'xnat_{:0>3d}_{:0>3d}_img.npy'.format(mAs, file_ind)
                            np.save(os.path.join(args.save_root, f_name), f.astype(np.uint16))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/your data save path/xnat/28272188/scans/')   # data format: dicom
    parser.add_argument('--save_root', type=str, default='./gen_data/xnat_npy/')
    args = parser.parse_args()

    save_dataset(args)