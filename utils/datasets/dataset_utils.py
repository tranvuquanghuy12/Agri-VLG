from utils.datasets.imagenet_1k import ImageNet1K
from utils.datasets.inat_dataset import iNatDataset
from torchvision.datasets import Flowers102
from torch.utils.data import Dataset
from PIL import Image
import PIL.Image
import pathlib
import os
import torch
from torchvision.datasets import folder as dataset_parser
import numpy as np
try:
    import tifffile
except ImportError:
    tifffile = None

NUM_CLASSES_DICT = {
    'semi-aves': 200,
    'flowers102': 102,
    'fgvc-aircraft': 100,
    'eurosat': 10,
    'eurosat_ms': 10,
    'dtd': 47,
    'food101': 101,
    'stanford_cars': 196,
    "oxford_pets": 37,
    'imagenet': 1000,
    'semi-inat-2021': 810,
    'tlu_states': 23,
}

def eurosat_ms_loader(path):
    if tifffile is None:
        return dataset_parser.default_loader(path)
    try:
        img = tifffile.imread(path)
        rgb = img[:, :, [3, 2, 1]]
        rgb = rgb.astype(np.float32)
        for i in range(3):
            ch = rgb[:, :, i]
            mi, ma = ch.min(), ch.max()
            if ma > mi: rgb[:, :, i] = (ch - mi) / (ma - mi) * 255.0
            else: rgb[:, :, i] = 0
        rgb = rgb.astype(np.uint8)
        return Image.fromarray(rgb)
    except Exception:
        return dataset_parser.default_loader(path)

def load_dataset(dataset_root, split, preprocess, tokenized_text_prompts, pl_list=None):
    dataset = MyDataset(dataset_root=dataset_root, split=split, transform=preprocess, tokenized_text_prompts=tokenized_text_prompts)
    return dataset

class SemiAvesDataset(Dataset):
    def __init__(self, dataset_root, split, transform, tokenized_text_prompts, loader=dataset_parser.default_loader):
        self.dataset_root, self.loader, self.transform, self.tokenized_text_prompts = pathlib.Path(dataset_root), loader, transform, tokenized_text_prompts
        self.data, self.labels = [], []
        with open(os.path.join(self.dataset_root, split), 'r') as f:
            for line in f:
                p, i, s = line.strip('\n').split(' ')
                self.data.append((os.path.join(self.dataset_root, p), int(i), int(s)))
                self.labels.append(int(i))
        self.targets = self.labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        try:
            img = self.loader(self.data[i][0])
            img = self.transform(img)
        except: img = torch.zeros(3, 224, 224)
        l, s = self.data[i][1], self.data[i][2]
        t = self.tokenized_text_prompts[str(l)]['all']
        t = t[torch.randint(0, t.size(0), (1,))]
        return img, l, t, s

class MyDataset(Dataset):
    def __init__(self, dataset_root, split, transform, tokenized_text_prompts, loader=None):
        self.dataset_root = pathlib.Path(dataset_root)
        self.loader = loader if loader else (eurosat_ms_loader if 'eurosat_ms' in str(dataset_root).lower() else dataset_parser.default_loader)
        fl, pl, lines = split[0], split[1], []
        for f_name, p_name in zip(fl, pl):
            with open(os.path.join(self.dataset_root, f_name), 'r') as f:
                for l in f:
                    l = l.strip()
                    if not l: continue
                    p = l.split(' ')
                    if len(p) < 3: continue 
                    s, lid, ip = p[-1], p[-2], ' '.join(p[:-2])
                    if os.path.isabs(ip):
                        if "/scratch/group/real-fs/retrieved/semi-aves/" in ip: ip = ip.replace("/scratch/group/real-fs/retrieved/semi-aves/", str(p_name) + "/").replace("//", "/")
                    else:
                        cp = p_name.replace('./', '').replace('.\\', '')
                        if not (ip.startswith(p_name) or ip.startswith(cp)): ip = os.path.join(p_name, ip)
                    lines.append((ip, int(lid), int(s)))
        self.data, self.labels, self.transform, self.tokenized_text_prompts = lines, [it[1] for it in lines], transform, tokenized_text_prompts
        self.targets = self.labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        try:
            img = self.loader(self.data[i][0])
            img = self.transform(img)
        except: img = torch.zeros(3, 224, 224)
        l, s = self.data[i][1], self.data[i][2]
        t = self.tokenized_text_prompts[str(l)]['all']
        t = t[torch.randint(0, t.size(0), (1,))]
        return img, l, t, s

class MyUnlabeledDataset(Dataset):
    def __init__(self, dataset_root, split, transform, loader=dataset_parser.default_loader):
        self.dataset_root, self.loader, self.transform = pathlib.Path(dataset_root), loader, transform
        fl, pl, lines = split[0], split[1], []
        for f_name, p_name in zip(fl, pl):
            with open(os.path.join(self.dataset_root, f_name), 'r') as f:
                for l in f: lines.append(os.path.join(p_name, l.strip('\n')))
        self.data, self.labels = [], []
        for line in lines:
            p = line.strip('\n').split(' ')
            if len(p) == 3: fp, lid, s = p
            else: s, lid, fp = p[-1], p[-2], ' '.join(p[:-2])
            self.data.append((fp, int(lid), int(s)))
            self.labels.append(int(lid))
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        try:
            img = self.loader(self.data[i][0])
            img = self.transform(img)
        except: img = torch.zeros(3, 224, 224)
        return img, self.data[i][1], torch.zeros(1, 1).long(), self.data[i][2]

class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, pre_extracted_path=None, device='cuda:0'):
        d = torch.load(pre_extracted_path, map_location=device)
        self.i, self.l = d['image_features'], d['labels']
    def __getitem__(self, index): return self.i[index], self.l[index], self.l[index], -1
    def __len__(self): return self.i.size(0)

class TextTensorDataset(torch.utils.data.Dataset):
    def __init__(self, prompt_tensors, device):
        ll, pl = [], []
        for cid, info in prompt_tensors.items():
            ll.append(torch.Tensor([int(cid)] * int(info['all'].shape[0])).long())
            pl.append(prompt_tensors[cid]['all'])
        self.l, self.p = torch.cat(ll).flatten().to(device), torch.cat(pl, dim=0).to(device)
    def __getitem__(self, i): return self.p[i], self.l[i]
    def __len__(self): return self.p.size(0)

class MinedDataset(Dataset):
    def __init__(self, transform=None, dataset_root=".", caption_map=None):
        self.root = pathlib.Path(dataset_root)
        self.fnames = list(self.root.glob("**/*.jpg")) + list(self.root.glob("**/*.tif"))
        self.caption_map, self.transform = caption_map, transform
        # Create a label map for string folder names
        all_folders = sorted(list(set([f.parent.name for f in self.fnames])))
        self.label_map = {name: i for i, name in enumerate(all_folders)}
    def __len__(self): return len(self.fnames)
    def __getitem__(self, i):
        p = str(self.fnames[i])
        try:
            img = eurosat_ms_loader(p) if p.endswith('.tif') else Image.open(p)
            if self.transform: img = self.transform(img)
        except: img = torch.zeros(3, 224, 224)
        fn, iid = self.fnames[i].parent.name, self.fnames[i].name.split('.')[0]
        try:
            label = int(fn)
        except:
            label = self.label_map[fn]
        c = self.caption_map[str(label)][iid] if self.caption_map else ""
        return img, label, p, c
