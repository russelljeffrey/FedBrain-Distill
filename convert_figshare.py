import h5py
import numpy as np
import cv2


def extract_data_from_hdf5(item):
    if isinstance(item, h5py.Dataset):
        return item[()]
    elif isinstance(item, h5py.Group):
        data_dict = {}
        for key in item.keys():
            data_dict[key] = extract_data_from_hdf5(item[key])
        return data_dict
    else:
        raise TypeError("Unsupported HDF5 item type.")

def load_hdf5_mat_to_numpy(file_path: str):
    with h5py.File(file_path, 'r') as file:
        keys = list(file.keys())
        if not keys:
            raise KeyError('No valid data keys found in the .mat file.')
        data_key = keys[0]
        data = extract_data_from_hdf5(file[data_key])
        return data

def convert_mat_to_numpy(data_dir: str = "figshare_data/", num_files: int = 3064):
    """
    This is a function that converts all .mat files to numpy arrays.
    This function uses load_hdf5_mat_to_numpy and extract_data_from_hdf5 functions.

    Args:
        data_dir (str): this is the directory of figshare .mat images.
        num_files (int): this is the number of files that exist in this directory.

    Returns:
        3 numpy arrays called images, labels, masks, which are tumor images, tumor labels and tumor region masks respectively.

    """
    images = []
    labels = []
    masks = []
    
    for i in range(num_files):
        data = load_hdf5_mat_to_numpy(f"{data_dir}/{i+1}.mat")
        image = data["image"]
        label = data["label"]
        mask = data["tumorMask"]

        image = image.astype("uint16")
        images.append(image)

        label = int(label[0][0])
        labels.append([label])
        masks.append(mask)

    height = 224
    width = 224
    for i in range(len(images)):
        if images[i].shape != (224, 224):
            images[i] = cv2.resize(images[i], (height, width), interpolation=cv2.INTER_AREA)

        if masks[i].shape != (224, 224):
            masks[i] = cv2.resize(masks[i], (height, width), interpolation=cv2.INTER_NEAREST)

    images = np.stack(images)
    labels = np.array(labels)
    masks = np.stack(masks)

    return images, labels, masks