"""
Adapted from
https://gitlab.lrz.de/CAMP_IFL/dynamic-dataset/dynamic-dataset/dynamic_dataset/datasets/dynamic/utils.py
See https://gitlab.lrz.de/CAMP_IFL/dynamic-dataset/-/tree/master
on how to install and use the dynamic dataset.
"""
from pathlib import Path
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
import hashlib
import torch
import torch.nn.init as init


def load_config_yaml(path):
    """loads a yaml config from file and returns a dict"""
    path = Path(path)
    with open(path) as file:
        cfg = yaml.full_load(file)
    return cfg


def save_config_yaml(path, config):
    path = Path(path)
    with open(path, "w") as file:
        yaml.dump(config, file)


def rm_tree(pth: Path):
    """WARNING: deletes path recursively like rm -rf"""
    print(f"Recursively deleting '{pth}'")
    for child in pth.iterdir():
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    pth.rmdir()


def get_sha256_hash(path):
    """returns sha256 hash from file found at path"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_idxs(idxs_in, test_size=0.1, val_size=0.1, seed=42, shuffle=True):
    """split indices into test, val and train
    """
    idxs_out = {}
    if test_size > 0:
        idxs_out["train"], idxs_out["test"] = train_test_split(
            idxs_in, test_size=test_size, shuffle=shuffle, stratify=None, random_state=seed
        )
    else:
        idxs_out["test"] = []
        idxs_out["train"] = idxs_in
    if val_size > 0:
        idxs_out["train"], idxs_out["val"] = train_test_split(
            idxs_out["train"],
            test_size=val_size / (1 - test_size),
            shuffle=True,
            stratify=None,
            random_state=seed,
        )
    else:
        idxs_out["val"] = []
    return idxs_out


def save_hash(hash, path):
    """save hash to given path"""
    with open(path, "w") as hash_file:
        print(hash, file=hash_file, end="")


def load_hash(path):
    """load hash from path"""
    with open(path, "r") as hash_file:
        return hash_file.read()


def verify_config_hash(config_path, npy_path: Path):
    """checks if config is the same as hashed and return bool"""
    hash_path = npy_path / "config_hash.sha256"
    if hash_path.is_file():
        new_hash = get_sha256_hash(config_path)
        old_hash = load_hash(hash_path)
        if new_hash == old_hash:
            return True
    return False


def save_config_hash(config_path, npy_path: Path):
    """saves hash of given config"""
    cfg_hash = get_sha256_hash(config_path)
    hash_path = npy_path / "config_hash.sha256"
    save_hash(cfg_hash, hash_path)


def crop_to_mask(data, seg, lung, crop_threshold=-1000000000):
    """
        crop data and return non-zero mask
        inspired by nnunet and stackoverflow
        """
    mask = np.zeros(data.shape, dtype=bool)
    # non zero mask over all channels
    cmask = data > crop_threshold
    mask = cmask | mask
    # non black coordinates
    coords = np.argwhere(mask)
    # bounding box
    x_min, y_min, z_min = coords.min(axis=0)
    x_max, y_max, z_max = coords.max(axis=0) + 1  # include top slice
    # crop each channel

    cropped_data = data[x_min:x_max, y_min:y_max, z_min:z_max]
    cropped_seg = seg[x_min:x_max, y_min:y_max, z_min:z_max]
    cropped_lung = lung[x_min:x_max, y_min:y_max, z_min:z_max]
    mask = mask[x_min:x_max, y_min:y_max, z_min:z_max]

    coords = np.argwhere(cropped_seg)
    coords2 = np.argwhere(cropped_lung)
    # bounding box

    x_min, y_min, z_min = np.concatenate((np.array([coords2.min(axis=0)]), np.array([coords.min(axis=0)])), axis=0).min(
        axis=0)  # change to : 'coords2.min(axis=0)' for only considering lung mask
    x_max, y_max, z_max = np.concatenate((np.array([coords2.max(axis=0)]), np.array([coords.max(axis=0)])), axis=0).max(
        axis=0) + 1  # include top slice # change to: 'coords2.max(axis=0)' for only considering lung mask

    cropped_lung = cropped_lung[x_min:x_max, y_min:y_max, z_min:z_max]
    cropped_seg = cropped_seg[x_min:x_max, y_min:y_max, z_min:z_max]
    cropped_mask = mask[x_min:x_max, y_min:y_max, z_min:z_max]

    cropped_data = cropped_data[x_min:x_max, y_min:y_max, z_min:z_max]



    return np.array(cropped_data), np.array(cropped_seg), np.array(cropped_lung), cropped_mask


def movedim(tensor: torch.Tensor, source: int, destination: int) -> torch.Tensor:
    dim = tensor.dim()
    perm = list(range(dim))
    if destination < 0:
        destination += dim
    perm.pop(source)
    perm.insert(destination, source)
    return tensor.permute(*perm)


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    # print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [%s] is not implemented' % init_type)
