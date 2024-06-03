import ast
import io
import logging
import os
import pdb
import torchvision
import torch
from PIL import Image, ImageFile

from .base_dataset import BaseDataset

# to fix "OSError: image file is truncated"
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImglistDataset(torch.utils.data.Dataset):
    def __init__(self, imglist_pth, data_dir, transform=None):
        with open(imglist_pth) as imgfile:
            self.imglist = imgfile.readlines()
        self.transform = transform
        self.data_dir = data_dir

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, idx):

        line = self.imglist[idx].strip('\n')
        tokens = line.split(' ', 1)
        image_name, extra_str = tokens[0], tokens[1]
        if self.data_dir != '' and image_name.startswith('/'):
            raise RuntimeError('image_name starts with "/"')
        img_path = os.path.join(self.data_dir, image_name)
        image_tensor = torchvision.io.read_image(img_path)
        image_pil = torchvision.transforms.ToPILImage()(image_tensor)
        if self.transform:
            image_tensor = torchvision.transforms.ToTensor()(image_pil)
        return image_tensor, int(extra_str)

'''

class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


class ImglistDataset(BaseDataset):
    def __init__(self,
                 imglist_pth,
                 data_dir,
                 preprocessor,
                 #data_aux_preprocessor,
                 maxlen=None,
                 dummy_read=False,
                 dummy_size=None,
                 **kwargs):
        super(ImglistDataset, self).__init__(**kwargs)

        #self.name = name
        with open(imglist_pth) as imgfile:
            self.imglist = imgfile.readlines()
        self.data_dir = data_dir
        #self.num_classes = num_classes
        self.preprocessor = preprocessor
        self.transform_image = preprocessor
        #self.transform_aux_image = data_aux_preprocessor
        self.maxlen = maxlen
        self.dummy_read = dummy_read
        self.dummy_size = dummy_size
        if dummy_read and dummy_size is None:
            raise ValueError(
                'if dummy_read is True, should provide dummy_size')

    def __len__(self):
        if self.maxlen is None:
            return len(self.imglist)
        else:
            return min(len(self.imglist), self.maxlen)

    def getitem(self, index):
        line = self.imglist[index].strip('\n')
        tokens = line.split(' ', 1)
        image_name, extra_str = tokens[0], tokens[1]
        if self.data_dir != '' and image_name.startswith('/'):
            raise RuntimeError('image_name starts with "/"')
        path = os.path.join(self.data_dir, image_name)
        sample = dict()
        sample['image_name'] = image_name
        #kwargs = {'name': self.name, 'path': path, 'tokens': tokens}
        try:
            # some preprocessor methods require setup
            self.preprocessor.setup(**kwargs)
        except:
            pass

        try:
            if not self.dummy_read:
                with open(path, 'rb') as f:
                    content = f.read()
                filebytes = content
                buff = io.BytesIO(filebytes)
            if self.dummy_size is not None:
                sample['data'] = torch.rand(self.dummy_size)
            else:
                image = Image.open(buff).convert('RGB')
                sample['data'] = self.transform_image(image)
                #sample['data_aux'] = self.transform_aux_image(image)
            extras = ast.literal_eval(extra_str)
            try:
                for key, value in extras.items():
                    sample[key] = value
                # if you use dic the code below will need ['label']
                sample['label'] = 0
            except AttributeError:
                sample['label'] = int(extra_str)

        except Exception as e:
            logging.error('[{}] broken'.format(path))
            raise e
        return sample['data'], sample['label']
    '''

