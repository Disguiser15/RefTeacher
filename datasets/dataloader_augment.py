# coding=utf-8
# Copyright 2022 The SimREC Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import cv2
import json, re, en_vectors_web_lg, random

import albumentations as A
import numpy as np

import torch
import torch.utils.data as Data
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from utils.utils import label2yolobox
from utils.box_ops import box_xywh_to_xyxy,box_cxcywh_to_xyxy
import datasets.transforms as T
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image

class RefCOCODataSet(Data.Dataset):
    def __init__(self, __C,split):
        super(RefCOCODataSet, self).__init__()
        self.__C = __C
        self.split=split
        assert  __C.DATASET in ['refcoco', 'refcoco+', 'refcocog','referit','vg','merge']
        # --------------------------
        # ---- Raw data loading ---
        # --------------------------
        stat_refs_list=json.load(open(__C.ANN_PATH[__C.DATASET], 'r'))
        total_refs_list=[]
        if __C.DATASET in ['vg','merge']:
            total_refs_list = json.load(open(__C.ANN_PATH['merge'], 'r'))+json.load(open(__C.ANN_PATH['refcoco+'], 'r'))+json.load(open(__C.ANN_PATH['refcocog'], 'r'))+json.load(open(__C.ANN_PATH['refcoco'], 'r'))

        self.ques_list = []
        splits=split.split('+')
        self.refs_anno=[]
        for split_ in splits:
            self.refs_anno+= stat_refs_list[split_]


        refs=[]

        for split in stat_refs_list:
            for ann in stat_refs_list[split]:
                for ref in ann['refs']:
                    refs.append(ref)
        for split in total_refs_list:
            for ann in total_refs_list[split]:
                for ref in ann['refs']:
                    refs.append(ref)



        self.image_path=__C.IMAGE_PATH[__C.DATASET]
        self.mask_path=__C.MASK_PATH[__C.DATASET]
        self.input_shape=__C.INPUT_SHAPE
        self.flip_lr=__C.FLIP_LR if split=='train' else False
        # Define run data size
        self.data_size = len(self.refs_anno)

        print(' ========== Dataset size:', self.data_size)
        # ------------------------
        # ---- Data statistic ----
        # ------------------------
        # Tokenize
        self.token_to_ix,self.ix_to_token, self.pretrained_emb, max_token = self.tokenize(stat_refs_list, __C.USE_GLOVE)
        self.token_size = self.token_to_ix.__len__()
        print(' ========== Question token vocab size:', self.token_size)

        self.max_token = __C.MAX_TOKEN
        if self.max_token == -1:
            self.max_token = max_token
        print('Max token length:', max_token, 'Trimmed to:', self.max_token)
        print('Finished!')
        print('')

        self.transforms=make_transforms(__C,self.split)

            # if 'RandAugment' in self.__C.DATA_AUGMENTATION:
            #     self.candidate_transforms['RandAugment']=RandAugment(2,9)
            # if 'ElasticTransform' in self.__C.DATA_AUGMENTATION:
            #     self.candidate_transforms['ElasticTransform']=A.ElasticTransform(p=0.5)
            # if 'GridDistortion' in self.__C.DATA_AUGMENTATION:
            #     self.candidate_transforms['GridDistortion']=A.GridDistortion(p=0.5)
            # if 'RandomErasing' in self.__C.DATA_AUGMENTATION:
            #     self.candidate_transforms['RandomErasing']=transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.05, 8),
            #                                                                   value="random")
        # self.transforms=transforms.Compose([transforms.ToTensor(), transforms.Normalize(__C.MEAN, __C.STD)])



    def tokenize(self, stat_refs_list, use_glove):
        token_to_ix = {
            'PAD': 0,
            'UNK': 1,
            'CLS': 2,
        }

        spacy_tool = None
        pretrained_emb = []
        if use_glove:
            spacy_tool = en_vectors_web_lg.load()
            pretrained_emb.append(spacy_tool('PAD').vector)
            pretrained_emb.append(spacy_tool('UNK').vector)
            pretrained_emb.append(spacy_tool('CLS').vector)

        max_token = 0
        for split in stat_refs_list:
            for ann in stat_refs_list[split]:
                for ref in ann['refs']:
                    words = re.sub(
                        r"([.,'!?\"()*#:;])",
                        '',
                        ref.lower()
                    ).replace('-', ' ').replace('/', ' ').split()

                    if len(words) > max_token:
                        max_token = len(words)

                    for word in words:
                        if word not in token_to_ix:
                            token_to_ix[word] = len(token_to_ix)
                            if use_glove:
                                pretrained_emb.append(spacy_tool(word).vector)

        pretrained_emb = np.array(pretrained_emb)
        ix_to_token={}
        for item in token_to_ix:
            ix_to_token[token_to_ix[item]]=item

        return token_to_ix, ix_to_token,pretrained_emb, max_token


    def proc_ref(self, ref, token_to_ix, max_token):
        ques_ix = np.zeros(max_token, np.int64)

        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ref.lower()
        ).replace('-', ' ').replace('/', ' ').split()

        for ix, word in enumerate(words):
            if word in token_to_ix:
                ques_ix[ix] = token_to_ix[word]
            else:
                ques_ix[ix] = token_to_ix['UNK']

            if ix + 1 == max_token:
                break

        return ques_ix

    # ----------------------------------------------
    # ---- Real-Time Processing Implementations ----
    # ----------------------------------------------

    def load_refs(self, idx):
        refs = self.refs_anno[idx]['refs']
        ref=refs[np.random.choice(len(refs))]
        return ref

    def preprocess_info(self,img,mask,box,iid,lr_flip=False):
        h, w, _ = img.shape
        # img = img[:, :, ::-1]
        imgsize=self.input_shape[0]
        new_ar = w / h
        if new_ar < 1:
            nh = imgsize
            nw = nh * new_ar
        else:
            nw = imgsize
            nh = nw / new_ar
        nw, nh = int(nw), int(nh)


        dx = (imgsize - nw) // 2
        dy = (imgsize - nh) // 2

        img = cv2.resize(img, (nw, nh))
        sized = np.ones((imgsize, imgsize, 3), dtype=np.uint8) * 127
        sized[dy:dy + nh, dx:dx + nw, :] = img
        info_img = (h, w, nh, nw, dx, dy,iid)

        mask=np.expand_dims(mask,-1).astype(np.float32)
        mask=cv2.resize(mask, (nw, nh))
        mask=np.expand_dims(mask,-1).astype(np.float32)
        sized_mask = np.zeros((imgsize, imgsize, 1), dtype=np.float32)
        sized_mask[dy:dy + nh, dx:dx + nw, :]=mask
        sized_mask=np.transpose(sized_mask, (2, 0, 1))
        sized_box=label2yolobox(box,info_img,self.input_shape[0],lrflip=lr_flip)
        return sized,sized_mask,sized_box, info_img

    def load_img_feats(self, idx):
        img_path=None
        if self.__C.DATASET in ['refcoco','refcoco+','refcocog']:
            img_path=os.path.join(self.image_path,'COCO_train2014_%012d.jpg'%self.refs_anno[idx]['iid'])
        elif self.__C.DATASET=='referit':
            img_path = os.path.join(self.image_path, '%d.jpg' % self.refs_anno[idx]['iid'])
        elif self.__C.DATASET=='vg':
            img_path = os.path.join(self.image_path, self.refs_anno[idx]['url'])
        elif self.__C.DATASET == 'merge':
            if self.refs_anno[idx]['data_source']=='coco':
                iid='COCO_train2014_%012d.jpg'%int(self.refs_anno[idx]['iid'].split('.')[0])
            else:
                iid=self.refs_anno[idx]['iid']
            img_path = os.path.join(self.image_path,self.refs_anno[idx]['data_source'], iid)
        else:
            assert NotImplementedError

        image= Image.open(img_path).convert('RGB') #cv2.imread(img_path)
        # print(image.size)
        if self.__C.DATASET in ['refcoco','refcoco+','refcocog','referit']:
            mask=np.load(os.path.join(self.mask_path,'%d.npy'%self.refs_anno[idx]['mask_id']))
        else:
            mask=np.zeros([image.shape[0],image.shape[1],1],dtype=np.float)

        box=np.array(self.refs_anno[idx]['bbox'])
        mask=Image.fromarray(mask*255)

        # print(box)
        return image,mask,box,self.refs_anno[idx]['mask_id'],self.refs_anno[idx]['iid']

    def __getitem__(self, idx):

        ref_iter = self.load_refs(idx)
        image_iter,mask_iter,gt_box_iter,mask_id,iid= self.load_img_feats(idx)
        w,h=image_iter.size
        # print((torch.from_numpy(gt_box_iter).float()))
        #left,top,w,h-> xyxy
        input_dict={'img':image_iter,
                    'box':box_xywh_to_xyxy(torch.from_numpy(gt_box_iter).float()),
                    'mask':mask_iter,
                    'text':ref_iter}
        # print(input_dict['text'])
        input_dict=self.transforms(input_dict)
        # print(input_dict['box']*self.input_shape[0])
        ref_iter=self.proc_ref(input_dict['text'],self.token_to_ix,self.max_token)

        info_iter=[h,w,*input_dict['info_img'],iid]

        # image_iter, mask_iter, box_iter,info_iter=self.preprocess_info(image_iter,mask_iter,gt_box_iter.copy(),iid,flip_box)
        return \
            torch.from_numpy(ref_iter).long(), \
            input_dict['img'], \
            input_dict['mask'], \
            input_dict['box'], \
            torch.from_numpy(gt_box_iter).float(), \
            mask_id,\
            np.array(info_iter)

    def __len__(self):
        return self.data_size

    def shuffle_list(self, list):
        random.shuffle(list)

def make_transforms(__C, image_set,label=True):
    # if is_onestage:
    #     normalize = Compose([
    #         ToTensor(),
    #         Normalize(__C.MEAN, __C.STD)
    #     ])
    #     return normalize

    imsize = __C.INPUT_SHAPE[0]

    if image_set == 'train':
        scales = []
        if __C.AUG_SCALE:
            # scales=[256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512, 528, 544, 560, 576, 592, 608]
            for i in range(7):
                scales.append(imsize - 32 * i)
        else:
            scales = [imsize]

        if __C.AUG_CROP:
            crop_prob = 0.5
        else:
            crop_prob = 0.

        # # q:randomresize+horizontalflip+colorjitter+aug_translate w:randomresize+horizontalflip+aug_translate
        # if label==False:
        #     return T.Compose([
        #         T.RandomSelect(
        #             T.RandomResize(scales),
        #             T.Compose([
        #                 T.RandomResize([400, 500, 600], with_long_side=False),
        #                 T.RandomSizeCrop(384, 600),
        #                 T.RandomResize(scales),
        #             ]),
        #             p=crop_prob
        #         ), 
        #         T.RandomHorizontalFlip()
        #         ]), T.Compose([
        #         T.ToTensor(),
        #         T.NormalizeAndPad(mean=__C.MEAN,std=__C.STD,size=imsize, aug_translate=__C.AUG_TRANSLATE)
        #     ]), T.Compose([
        #         T.ColorJitter(0.4, 0.4, 0.4),
        #         T.GaussianBlur(aug_blur=__C.AUG_BLUR),
        #         T.ToTensor(),
        #         T.NormalizeAndPad(mean=__C.MEAN,std=__C.STD,size=imsize, aug_translate=__C.AUG_TRANSLATE)
        #     ])

        # else:
        #     return T.Compose([
        #         T.RandomSelect(
        #             T.RandomResize(scales),
        #             T.Compose([
        #                 T.RandomResize([400, 500, 600], with_long_side=False),
        #                 T.RandomSizeCrop(384, 600),
        #                 T.RandomResize(scales),
        #             ]),
        #             p=crop_prob
        #         ),
        #         T.ColorJitter(0.4, 0.4, 0.4),
        #         T.GaussianBlur(aug_blur=__C.AUG_BLUR),
        #         T.RandomHorizontalFlip(),
        #         T.ToTensor(),
        #         T.NormalizeAndPad(mean=__C.MEAN,std=__C.STD,size=imsize, aug_translate=__C.AUG_TRANSLATE)
        #     ])
        
        # q:randomresize+horizontalflip+colorjitter+aug_translate w:horizontalflip
        if label==False:
            return T.Compose([
                T.RandomResize([imsize]),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.NormalizeAndPad(size=imsize),
            ]), T.Compose([
                T.RandomSelect(
                    T.RandomResize(scales),
                    T.Compose([
                        T.RandomResize([400, 500, 600], with_long_side=False),
                        T.RandomSizeCrop(384, 600),
                        T.RandomResize(scales),
                    ]),
                    p=crop_prob
                ),
                T.ColorJitter(0.4, 0.4, 0.4),
                T.GaussianBlur(aug_blur=__C.AUG_BLUR),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.NormalizeAndPad(size=imsize, aug_translate=__C.AUG_TRANSLATE)
            ])

        else:
            return T.Compose([
                T.RandomSelect(
                    T.RandomResize(scales),
                    T.Compose([
                        T.RandomResize([400, 500, 600], with_long_side=False),
                        T.RandomSizeCrop(384, 600),
                        T.RandomResize(scales),
                    ]),
                    p=crop_prob
                ),
                T.ColorJitter(0.4, 0.4, 0.4),
                T.GaussianBlur(aug_blur=__C.AUG_BLUR),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.NormalizeAndPad(mean=__C.MEAN,std=__C.STD,size=imsize, aug_translate=__C.AUG_TRANSLATE)
            ])

        # # q:randomresize w:noun
        # if label==False:
        #     return T.Compose([
        #         ]), T.Compose([
        #         T.RandomResize([imsize]),
        #         T.ToTensor(),
        #         T.NormalizeAndPad(mean=__C.MEAN,std=__C.STD,size=imsize, aug_translate=__C.AUG_TRANSLATE)
        #     ]), T.Compose([
        #         T.RandomResize(scales),
        #         T.ToTensor(),
        #         T.NormalizeAndPad(mean=__C.MEAN,std=__C.STD,size=imsize, aug_translate=__C.AUG_TRANSLATE)
        #     ])

        # else:
        #     return T.Compose([
        #         T.RandomSelect(
        #             T.RandomResize(scales),
        #             T.Compose([
        #                 T.RandomResize([400, 500, 600], with_long_side=False),
        #                 T.RandomSizeCrop(384, 600),
        #                 T.RandomResize(scales),
        #             ]),
        #             p=crop_prob
        #         ),
        #         # T.ColorJitter(0.4, 0.4, 0.4),
        #         # T.GaussianBlur(aug_blur=__C.AUG_BLUR),
        #         # T.RandomHorizontalFlip(),
        #         T.ToTensor(),
        #         T.NormalizeAndPad(mean=__C.MEAN,std=__C.STD,size=imsize, aug_translate=__C.AUG_TRANSLATE)
        #     ])

    if image_set in ['val', 'test', 'testA', 'testB']:
        return T.Compose([
            T.RandomResize([imsize]),
            T.ToTensor(),
            T.NormalizeAndPad(mean=__C.MEAN,std=__C.STD,size=imsize),
        ])

    raise ValueError(f'unknown {image_set}')

class RefCOCODataSet_semi(Data.Dataset):
    def __init__(self, __C,split,sup=True,label=True):
        super(RefCOCODataSet_semi, self).__init__()
        self.__C = __C
        self.split=split
        self.label=label
        if sup == True:
            assert  __C.DATASET in ['refcoco_label', 'refcoco_unlabel', 'refcoco+_label', 'refcoco+_unlabel', 'refcocog_label', 'refcocog_unlabel', 'refcoco', 'refcoco+', 'refcocog','referit','vg','merge']
        else:
            assert  __C.DATASET_LABEL in ['refcoco_label', 'refcoco_unlabel', 'refcoco+_label', 'refcoco+_unlabel', 'refcocog_label', 'refcocog_unlabel', 'refcoco', 'refcoco+', 'refcocog','referit','vg','merge'] # ['refcoco_label', 'refcoco_unlabel', 'refcoco+_label', 'refcoco+_unlabel', 'refcocog_label', 'refcocog_unlabel', 'refcoco', 'refcoco+', 'refcocog','referit','vg','merge']
            assert  __C.DATASET_UNLABEL in ['refcoco_label', 'refcoco_unlabel', 'refcoco+_label', 'refcoco+_unlabel', 'refcocog_label', 'refcocog_unlabel', 'refcoco', 'refcoco+', 'refcocog','referit','vg','merge']
        # --------------------------
        # ---- Raw data loading ---
        # --------------------------
        if sup == True:
            stat_refs_list=json.load(open(__C.ANN_PATH[__C.DATASET], 'r'))
        else:
            stat_refs_list_label = stat_refs_list=json.load(open(__C.ANN_PATH[__C.DATASET_LABEL], 'r'))
            stat_refs_list_unlabel = stat_refs_list=json.load(open(__C.ANN_PATH[__C.DATASET_UNLABEL], 'r'))
            stat_refs_list_label_unlabel = dict()
            for key, value in stat_refs_list_label.items():
                if key == 'train':
                    stat_refs_list_label_unlabel[key] = stat_refs_list_label[key] + stat_refs_list_unlabel[key]
                else:
                    stat_refs_list_label_unlabel[key] = stat_refs_list_label[key]
        
        if sup == False and label == True:
            stat_refs_list=json.load(open(__C.ANN_PATH[__C.DATASET_LABEL], 'r'))
        elif sup==False and label == False:
            stat_refs_list=json.load(open(__C.ANN_PATH[__C.DATASET_UNLABEL], 'r'))
        total_refs_list=[]
        if __C.DATASET in ['vg','merge']:
            total_refs_list = json.load(open(__C.ANN_PATH['merge'], 'r'))+json.load(open(__C.ANN_PATH['refcoco+'], 'r'))+json.load(open(__C.ANN_PATH['refcocog'], 'r'))+json.load(open(__C.ANN_PATH['refcoco'], 'r'))

        self.ques_list = []
        splits=split.split('+')
        self.refs_anno=[]
        for split_ in splits:
            self.refs_anno+= stat_refs_list[split_]


        refs=[]

        for split in stat_refs_list:
            for ann in stat_refs_list[split]:
                for ref in ann['refs']:
                    refs.append(ref)
        for split in total_refs_list:
            for ann in total_refs_list[split]:
                for ref in ann['refs']:
                    refs.append(ref)



        self.image_path=__C.IMAGE_PATH[__C.DATASET]
        self.mask_path=__C.MASK_PATH[__C.DATASET]
        self.input_shape=__C.INPUT_SHAPE
        self.flip_lr=__C.FLIP_LR if split=='train' else False
        # Define run data size
        self.data_size = len(self.refs_anno)

        print(' ========== Dataset size:', self.data_size)
        # ------------------------
        # ---- Data statistic ----
        # ------------------------
        # Tokenize
        if sup == True:
            self.token_to_ix,self.ix_to_token, self.pretrained_emb, max_token = self.tokenize(stat_refs_list, __C.USE_GLOVE)
        else:
            self.token_to_ix,self.ix_to_token, self.pretrained_emb, max_token = self.tokenize(stat_refs_list_label_unlabel, __C.USE_GLOVE)
        self.token_size = self.token_to_ix.__len__()
        print(' ========== Question token vocab size:', self.token_size)

        self.max_token = __C.MAX_TOKEN
        if self.max_token == -1:
            self.max_token = max_token
        print('Max token length:', max_token, 'Trimmed to:', self.max_token)
        print('Finished!')
        print('')

        if label == False:
            self.transforms_w, self.transforms_q=make_transforms(__C,self.split, label=False)
        else:
            self.transforms=make_transforms(__C,self.split,label=True)

            # if 'RandAugment' in self.__C.DATA_AUGMENTATION:
            #     self.candidate_transforms['RandAugment']=RandAugment(2,9)
            # if 'ElasticTransform' in self.__C.DATA_AUGMENTATION:
            #     self.candidate_transforms['ElasticTransform']=A.ElasticTransform(p=0.5)
            # if 'GridDistortion' in self.__C.DATA_AUGMENTATION:
            #     self.candidate_transforms['GridDistortion']=A.GridDistortion(p=0.5)
            # if 'RandomErasing' in self.__C.DATA_AUGMENTATION:
            #     self.candidate_transforms['RandomErasing']=transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.05, 8),
            #                                                                   value="random")
        # self.transforms=transforms.Compose([transforms.ToTensor(), transforms.Normalize(__C.MEAN, __C.STD)])



    def tokenize(self, stat_refs_list, use_glove):
        token_to_ix = {
            'PAD': 0,
            'UNK': 1,
            'CLS': 2,
        }

        spacy_tool = None
        pretrained_emb = []
        if use_glove:
            spacy_tool = en_vectors_web_lg.load()
            pretrained_emb.append(spacy_tool('PAD').vector)
            pretrained_emb.append(spacy_tool('UNK').vector)
            pretrained_emb.append(spacy_tool('CLS').vector)

        max_token = 0
        for split in stat_refs_list:
            for ann in stat_refs_list[split]:
                for ref in ann['refs']:
                    words = re.sub(
                        r"([.,'!?\"()*#:;])",
                        '',
                        ref.lower()
                    ).replace('-', ' ').replace('/', ' ').split()

                    if len(words) > max_token:
                        max_token = len(words)

                    for word in words:
                        if word not in token_to_ix:
                            token_to_ix[word] = len(token_to_ix)
                            if use_glove:
                                pretrained_emb.append(spacy_tool(word).vector)

        pretrained_emb = np.array(pretrained_emb)
        ix_to_token={}
        for item in token_to_ix:
            ix_to_token[token_to_ix[item]]=item

        return token_to_ix, ix_to_token,pretrained_emb, max_token


    def proc_ref(self, ref, token_to_ix, max_token):
        ques_ix = np.zeros(max_token, np.int64)

        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ref.lower()
        ).replace('-', ' ').replace('/', ' ').split()

        for ix, word in enumerate(words):
            if word in token_to_ix:
                ques_ix[ix] = token_to_ix[word]
            else:
                ques_ix[ix] = token_to_ix['UNK']

            if ix + 1 == max_token:
                break

        return ques_ix

    # ----------------------------------------------
    # ---- Real-Time Processing Implementations ----
    # ----------------------------------------------

    def load_refs(self, idx):
        refs = self.refs_anno[idx]['refs']
        ref=refs[np.random.choice(len(refs))]
        return ref

    def preprocess_info(self,img,mask,box,iid,lr_flip=False):
        h, w, _ = img.shape
        # img = img[:, :, ::-1]
        imgsize=self.input_shape[0]
        new_ar = w / h
        if new_ar < 1:
            nh = imgsize
            nw = nh * new_ar
        else:
            nw = imgsize
            nh = nw / new_ar
        nw, nh = int(nw), int(nh)


        dx = (imgsize - nw) // 2
        dy = (imgsize - nh) // 2

        img = cv2.resize(img, (nw, nh))
        sized = np.ones((imgsize, imgsize, 3), dtype=np.uint8) * 127
        sized[dy:dy + nh, dx:dx + nw, :] = img
        info_img = (h, w, nh, nw, dx, dy,iid)

        mask=np.expand_dims(mask,-1).astype(np.float32)
        mask=cv2.resize(mask, (nw, nh))
        mask=np.expand_dims(mask,-1).astype(np.float32)
        sized_mask = np.zeros((imgsize, imgsize, 1), dtype=np.float32)
        sized_mask[dy:dy + nh, dx:dx + nw, :]=mask
        sized_mask=np.transpose(sized_mask, (2, 0, 1))
        sized_box=label2yolobox(box,info_img,self.input_shape[0],lrflip=lr_flip)
        return sized,sized_mask,sized_box, info_img

    def load_img_feats(self, idx):
        img_path=None
        if self.__C.DATASET in ['refcoco','refcoco+','refcocog']:
            img_path=os.path.join(self.image_path,'COCO_train2014_%012d.jpg'%self.refs_anno[idx]['iid'])
        elif self.__C.DATASET=='referit':
            img_path = os.path.join(self.image_path, '%d.jpg' % self.refs_anno[idx]['iid'])
        elif self.__C.DATASET=='vg':
            img_path = os.path.join(self.image_path, self.refs_anno[idx]['url'])
        elif self.__C.DATASET == 'merge':
            if self.refs_anno[idx]['data_source']=='coco':
                iid='COCO_train2014_%012d.jpg'%int(self.refs_anno[idx]['iid'].split('.')[0])
            else:
                iid=self.refs_anno[idx]['iid']
            img_path = os.path.join(self.image_path,self.refs_anno[idx]['data_source'], iid)
        else:
            assert NotImplementedError

        image= Image.open(img_path).convert('RGB') #cv2.imread(img_path)
        # print(image.size)
        if self.__C.DATASET in ['refcoco','refcoco+','refcocog','referit']:
            mask=np.load(os.path.join(self.mask_path,'%d.npy'%self.refs_anno[idx]['mask_id']))
        else:
            mask=np.zeros([image.shape[0],image.shape[1],1],dtype=np.float)

        box=np.array(self.refs_anno[idx]['bbox'])
        mask=Image.fromarray(mask*255)

        # print(box)
        return image,mask,box,self.refs_anno[idx]['mask_id'],self.refs_anno[idx]['iid']

    def __getitem__(self, idx):

        ref_iter = self.load_refs(idx)
        image_iter,mask_iter,gt_box_iter,mask_id,iid= self.load_img_feats(idx)
        w,h=image_iter.size
        # print((torch.from_numpy(gt_box_iter).float()))
        #left,top,w,h-> xyxy
        input_dict={'img':image_iter,
                    'box':box_xywh_to_xyxy(torch.from_numpy(gt_box_iter).float()),
                    'mask':mask_iter,
                    'text':ref_iter}
        # print(input_dict['text'])
        percent = random.random()
        if self.label == False:
            input_dict_w = input_dict.copy()
            input_dict_q = input_dict.copy()
            input_dict_w, percent_w = self.transforms_w(input_dict_w, percent) # weak augmentation
            input_dict_q_process, percent_q = self.transforms_q(input_dict_q, percent) # strong augmentation
            while input_dict_q_process['box'][:,0]==1 or input_dict_q_process['box'][:,1]==1 or input_dict_q_process['box'][:,2]==0 or input_dict_q_process['box'][:,3]==0:
                input_dict_copy = input_dict.copy()
                input_dict_q_process, percent=self.transforms_q(input_dict_copy, percent)

            ref_iter_w=self.proc_ref(input_dict_w['text'],self.token_to_ix,self.max_token)
            ref_iter_q=self.proc_ref(input_dict_q_process['text'],self.token_to_ix,self.max_token)
            info_iter_w=[h,w,*input_dict_w['info_img'],iid]
            info_iter_q=[h,w,*input_dict_q['info_img'],iid]
            return \
                torch.from_numpy(ref_iter_w).long(), \
                input_dict_w['img'], \
                input_dict_w['mask'], \
                input_dict_w['box'], \
                torch.from_numpy(gt_box_iter).float(), \
                mask_id,\
                np.array(info_iter_w), \
                torch.from_numpy(ref_iter_q).long(), \
                input_dict_q_process['img'], \
                input_dict_q_process['mask'], \
                input_dict_q_process['box'], \
                torch.from_numpy(gt_box_iter).float(), \
                mask_id,\
                np.array(info_iter_q)
        else:
            input_dict_original = input_dict.copy()
            input_dict_process, percent=self.transforms(input_dict, percent)
            while input_dict_process['box'][:,0]==1 or input_dict_process['box'][:,1]==1 or input_dict_process['box'][:,2]==0 or input_dict_process['box'][:,3]==0:
                input_dict_original_copy = input_dict_original.copy()
                input_dict_process, percent=self.transforms(input_dict_original_copy, percent)
            # print(input_dict['box']*self.input_shape[0])
            ref_iter=self.proc_ref(input_dict_process['text'],self.token_to_ix,self.max_token)

            info_iter=[h,w,*input_dict_process['info_img'],iid]

            # image_iter, mask_iter, box_iter,info_iter=self.preprocess_info(image_iter,mask_iter,gt_box_iter.copy(),iid,flip_box)
            return \
                torch.from_numpy(ref_iter).long(), \
                input_dict_process['img'], \
                input_dict_process['mask'], \
                input_dict_process['box'], \
                torch.from_numpy(gt_box_iter).float(), \
                mask_id,\
                np.array(info_iter)


    def __len__(self):
        return self.data_size

    def shuffle_list(self, list):
        random.shuffle(list)

class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers
    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()
        self.length=None

    def __len__(self):
        return len(self.batch_sampler.sampler) if self.length is None else max(len(self.batch_sampler.sampler),self.length)

    def __iter__(self):
        for i in range(len(self) if self.length is None else max(len(self),self.length)):
        # while True:
            yield next(self.iterator)
    def set_length(self,length):
        self.length=length


class _RepeatSampler(object):
    """ Sampler that repeats forever
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

def loader(__C,dataset: torch.utils.data.Dataset, rank: int, shuffle,drop_last=False):
    if __C.MULTIPROCESSING_DISTRIBUTED:
        assert __C.BATCH_SIZE % len(__C.GPU) == 0
        assert __C.NUM_WORKER % len(__C.GPU) == 0
        assert dist.is_initialized()

        dist_sampler = DistributedSampler(dataset,
                                          num_replicas=__C.WORLD_SIZE,
                                          rank=rank)

        data_loader = InfiniteDataLoader(dataset,
                                 batch_size=__C.BATCH_SIZE // len(__C.GPU),
                                 shuffle=shuffle,
                                 sampler=dist_sampler,
                                 num_workers=__C.NUM_WORKER //len(__C.GPU),
                                 pin_memory=False,
                                 drop_last=drop_last)  # ,
                                # prefetch_factor=_C['PREFETCH_FACTOR'])  only works in PyTorch 1.7.0
    else:
        data_loader = InfiniteDataLoader(dataset,
                                 batch_size=__C.BATCH_SIZE,
                                 shuffle=shuffle,
                                 num_workers=__C.NUM_WORKER,
                                 pin_memory=False,
                                 drop_last=drop_last)
    return data_loader

if __name__ == '__main__':

    class Cfg():
        def __init__(self):
            super(Cfg, self).__init__()
            self.ANN_PATH= {
                'refcoco': './data/anns/refcoco.json',
                'refcoco+': './data/anns/refcoco+.json',
                'refcocog': './data/anns/refcocog.json',
                'vg': './data/anns/vg.json',
            }

            self.IMAGE_PATH={
                'refcoco': './data/images/coco',
                'refcoco+': './data/images/coco',
                'refcocog': './data/images/coco',
                'vg': './data/images/VG'
            }

            self.MASK_PATH={
                'refcoco': './data/masks/refcoco',
                'refcoco+': './data/masks/refcoco+',
                'refcocog': './data/masks/refcocog',
                'vg': './data/masks/vg'}
            self.INPUT_SHAPE = (416, 416)
            self.USE_GLOVE = True
            self.DATASET = 'refcoco'
            self.MAX_TOKEN = 15
            self.MEAN = [0., 0., 0.]
            self.STD = [1., 1., 1.]
            self.AUG_SCALE=True
            self.AUG_CROP=True
            self.AUG_BLUR=False
            self.AUG_TRANSLATE=True
    cfg=Cfg()
    dataset=RefCOCODataSet(cfg,'train')
    data_loader = DataLoader(dataset,
                             batch_size=10,
                             shuffle=False,
                             pin_memory=True)
    for  ref_iter,image_iter,mask_iter,box_iter,gt_box_iter,mask_id,info_iter in data_loader:

        print(image_iter.size())
        print(mask_iter.size())
        # print(box_iter)
        # print(ref_iter.size())
        # print(info_iter)
        img=image_iter.numpy()[0].transpose((1, 2, 0))*255
        box_iter=box_cxcywh_to_xyxy(box_iter)[0,0]*416
        print(box_iter)
        box_iter=box_iter.long().numpy()
        img=np.ascontiguousarray(img,dtype=np.uint8)
        cv2.rectangle(img,(box_iter[0],box_iter[1]),(box_iter[2],box_iter[3]),(0,255,0),2)
        cv2.imwrite('./test.jpg', img)
        # cv2.imwrite('./mask.jpg', mask_iter.numpy()[0].transpose((1, 2, 0))*255)
        break
        # print(info_iter.size())
        # print(info_iter)





