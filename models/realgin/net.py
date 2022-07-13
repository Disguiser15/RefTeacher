import torch
import torch.nn as nn
from models.realgin.head import head
from models.language_encoder import language_encoder
from models.visual_encoder import visual_encoder
from layers.fusion_layer import SimpleFusion,MultiScaleFusion,AdaptiveFeatureSelection,GaranAttention



class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size):
        super(Net, self).__init__()
        self.visual_encoder=visual_encoder(__C)
        self.lang_encoder=language_encoder(__C,pretrained_emb,token_size)
        # self.afs=AdaptiveFeatureSelection(2,[256,512],0,[],1024,1024,256,512)
        self.afs=AdaptiveFeatureSelection(2,[256,512],0,[],1024,512,256,512)
        # self.garan=GaranAttention(1024,512,2)
        self.garan=GaranAttention(512,512,2)
        # self.fusion_manner=SimpleFusion(512)
        self.fusion_manner=SimpleFusion(v_planes=512,q_planes=512,out_planes=1024)
        self.head=head(__C,0,__C.HIDDEN_SIZE*2)
        total = sum([param.nelement() for param in self.lang_encoder.parameters()])
        print('  + Number of lang enc params: %.2fM' % (total / 1e6))  # 每一百万为一个单位

        total = sum([param.nelement() for param in self.afs.parameters()])
        total += sum([param.nelement() for param in self.fusion_manner.parameters()])
        total += sum([param.nelement() for param in self.garan.parameters()])
        total += sum([param.nelement() for param in self.head.parameters()])
        print('  + Number of fusion params: %.2fM' % (total / 1e6))  # 每一百万为一个单位
        if __C.VIS_FREEZE:
            if __C.VIS_ENC=='vgg' or __C.VIS_ENC=='darknet':
                self.frozen(self.visual_encoder.module_list[:-2])
            else:
                self.frozen(self.visual_encoder)
    def frozen(self,module):
        if getattr(module,'module',False):
            for child in module.module():
                for param in child.parameters():
                    param.requires_grad = False
        else:
            for param in module.parameters():
                param.requires_grad = False
    def forward(self, x,y, det_label=None,seg_label=None):
        x=self.visual_encoder(x)
        y=self.lang_encoder(y)
        x=self.afs(y['flat_lang_feat'],x)
        x,_,_=self.garan(y['flat_lang_feat'],x)
        x=self.fusion_manner(x,y['flat_lang_feat'])
        if self.training:
            loss,loss_det,loss_seg=self.head(x,None,det_label,seg_label)
            return loss,loss_det,loss_seg
        else:
            box, mask=self.head(x,None)
            return box,mask


if __name__ == '__main__':
    class Cfg():
        def __init__(self):
            super(Cfg, self).__init__()
            self.USE_GLOVE = False
            self.WORD_EMBED_SIZE = 300
            self.HIDDEN_SIZE = 512
            self.N_SA = 0
            self.FLAT_GLIMPSES = 8
            self.DROPOUT_R = 0.1
            self.LANG_ENC = 'lstm'
            self.VIS_ENC = 'darknet'
            self.VIS_PRETRAIN = False
            self.EMBED_FREEZE=False
            self.PRETTRAIN_WEIGHT = './darknet.weights'
            self.ANCHORS = [[116, 90], [156, 198], [373, 326]]
            self.ANCH_MASK = [[0, 1, 2]]
            self.N_CLASSES = 0
            self.VIS_FREEZE = True
    cfg=Cfg()
    model=Net(cfg,torch.zeros(1),100)
    model.eval()
    img=torch.zeros(2,3,224,224)
    lang=torch.randint(10,(2,14))
    seg, det=model(img,lang)
    print(seg.size(),det.size())