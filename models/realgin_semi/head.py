import torch
import torch.nn as nn
import numpy as np
from utils.utils import bboxes_iou
from layers.conv_layer import aspp_decoder

class head(nn.Module):
    """
    detection layer corresponding to yolo_layer.c of darknet
    """
    def __init__(self, __C, layer_no, in_ch, ignore_thre=0.5):
        """
        Args:
            config_model (dict) : model configuration.
                ANCHORS (list of tuples) :
                ANCH_MASK:  (list of int list): index indicating the anchors to be
                    used in YOLO layers. One of the mask group is picked from the list.
                N_CLASSES (int): number of classes
            layer_no (int): YOLO layer number - one from (0, 1, 2).
            in_ch (int): number of input channels.
            ignore_thre (float): threshold of IoU above which objectness training is ignored.
        """

        super(head, self).__init__()
        # strides = [32, 16, 8] # fixed
        self.anchors = __C.ANCHORS
        self.anch_mask = __C.ANCH_MASK[layer_no]
        self.n_anchors = len(self.anch_mask)
        self.n_classes = __C.N_CLASSES
        self.ignore_thre = ignore_thre
        self.l2_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.stride = 32#strides[layer_no]
        self.all_anchors_grid = [(w / self.stride, h / self.stride)
                                 for w, h in self.anchors]
        self.masked_anchors = [self.all_anchors_grid[i]
                               for i in self.anch_mask]
        self.ref_anchors = np.zeros((len(self.all_anchors_grid), 4))
        self.ref_anchors[:, 2:] = np.array(self.all_anchors_grid)
        self.ref_anchors = torch.FloatTensor(self.ref_anchors)
        self.dconv = nn.Conv2d(in_channels=in_ch,
                              out_channels=self.n_anchors * (self.n_classes + 5),
                              kernel_size=1, stride=1, padding=0)


    def forward(self, xin,yin, x_label=None,y_label=None,semi=False):
        """
        In this
        Args:
            xin (torch.Tensor): input feature map whose size is :math:`(N, C, H, W)`, \
                where N, C, H, W denote batchsize, channel width, height, width respectively.
            x_label (torch.Tensor): label data whose size is :math:`(N, K, 5)`. \
                N and K denote batchsize and number of x_label.
                Each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
        Returns:
            loss (torch.Tensor): total loss - the target of backprop.
            loss_xy (torch.Tensor): x, y loss - calculated by binary cross entropy (BCE) \
                with boxsize-dependent weights.
            loss_wh (torch.Tensor): w, h loss - calculated by l2 without size averaging and \
                with boxsize-dependent weights.
            loss_obj (torch.Tensor): objectness loss - calculated by BCE.
            loss_cls (torch.Tensor): classification loss - calculated by BCE for each class.
            loss_l2 (torch.Tensor): total l2 loss - only for logging.
        """
        output = self.dconv(xin)

        # mask=self.sconv(yin)
        # print(output.size(),mask.size())

        batchsize = output.shape[0]
        fsize = output.shape[2]
        n_ch = 5 + self.n_classes
        dtype = torch.cuda.FloatTensor if xin.is_cuda else torch.FloatTensor
        devices=xin.device

        output = output.view(batchsize, self.n_anchors, n_ch, fsize, fsize)
        output = output.permute(0, 1, 3, 4, 2).contiguous()

        # logistic activation for xy, obj, cls
        output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(
            output[..., np.r_[:2, 4:n_ch]])

        # calculate pred - xywh obj cls

        x_shift = dtype(np.broadcast_to(
            np.arange(fsize, dtype=np.float32), output.shape[:4])).to(devices)
        y_shift = dtype(np.broadcast_to(
            np.arange(fsize, dtype=np.float32).reshape(fsize, 1), output.shape[:4])).to(devices)

        masked_anchors = np.array(self.masked_anchors)

        w_anchors = dtype(np.broadcast_to(np.reshape(
            masked_anchors[:, 0], (1, self.n_anchors, 1, 1)), output.shape[:4])).to(devices)
        h_anchors = dtype(np.broadcast_to(np.reshape(
            masked_anchors[:, 1], (1, self.n_anchors, 1, 1)), output.shape[:4])).to(devices)

        pred = output.clone()
        pred[..., 0] += x_shift
        pred[..., 1] += y_shift
        pred[..., 2] = torch.exp(pred[..., 2]) * w_anchors
        pred[..., 3] = torch.exp(pred[..., 3]) * h_anchors

        if x_label is None:  # not training
            pred[..., :4] *= self.stride
            pred=pred.view(batchsize,-1,n_ch)
            #xc,yc,,w,h->xmin,ymin,xmax,ymax
            pred[:, :, 0] = pred[:, :, 0] - pred[:, :, 2] / 2
            pred[:, :, 1] = pred[:, :, 1] - pred[:, :, 3] / 2
            pred[:, :, 2] = pred[:, :, 0] + pred[:, :, 2]
            pred[:, :, 3] = pred[:, :, 1] + pred[:, :, 3]
            score=pred[:,:,4].sigmoid()
            ind=torch.argmax(score,-1).unsqueeze(1).unsqueeze(1).repeat(1,1,n_ch)
            pred=torch.gather(pred,1,ind)
            return pred.view(batchsize,-1),torch.zeros(batchsize,fsize*self.stride,fsize*self.stride).to(pred.device)

        pred = pred[..., :4].data

        # target assignment

        tgt_mask = torch.zeros(batchsize, self.n_anchors,
                               fsize, fsize, 4 + self.n_classes).type(dtype).to(devices)
        obj_mask = torch.ones(batchsize, self.n_anchors,
                              fsize, fsize).type(dtype).to(devices)
        tgt_scale = torch.zeros(batchsize, self.n_anchors,
                                fsize, fsize, 2).type(dtype).to(devices)

        target = torch.zeros(batchsize, self.n_anchors,
                             fsize, fsize, n_ch).type(dtype).to(devices)

        x_label = x_label.cpu().data
        nlabel = (x_label.sum(dim=2) > 0).sum(dim=1)  # number of objects

        truth_x_all = x_label[:, :, 0] * fsize
        truth_y_all = x_label[:, :, 1] * fsize
        truth_w_all = x_label[:, :, 2] * fsize
        truth_h_all = x_label[:, :, 3] * fsize
        truth_i_all = truth_x_all.to(torch.int16).numpy()
        truth_j_all = truth_y_all.to(torch.int16).numpy()

        for b in range(batchsize):
            n = int(nlabel[b])
            if n == 0:
                continue
            truth_box = dtype(np.zeros((n, 4))).to(devices)
            truth_box[:n, 2] = truth_w_all[b, :n]
            truth_box[:n, 3] = truth_h_all[b, :n]
            truth_i = truth_i_all[b, :n]
            truth_j = truth_j_all[b, :n]

            # calculate iou between truth and reference anchors
            anchor_ious_all = bboxes_iou(truth_box.cpu(), self.ref_anchors)
            best_n_all = np.argmax(anchor_ious_all, axis=1)
            best_n = best_n_all % 3
            best_n_mask = ((best_n_all == self.anch_mask[0]) | (
                best_n_all == self.anch_mask[1]) | (best_n_all == self.anch_mask[2]))

            truth_box[:n, 0] = truth_x_all[b, :n]
            truth_box[:n, 1] = truth_y_all[b, :n]

            pred_ious = bboxes_iou(
                pred[b].view(-1, 4), truth_box, xyxy=False)
            pred_best_iou, _ = pred_ious.max(dim=1)
            pred_best_iou = (pred_best_iou > self.ignore_thre)
            pred_best_iou = pred_best_iou.view(pred[b].shape[:3])
            # set mask to zero (ignore) if pred matches truth
            # obj_mask[b] = 1 - pred_best_iou
            obj_mask[b] = ~pred_best_iou

            if sum(best_n_mask) == 0:
                continue

            for ti in range(best_n.shape[0]):
                if best_n_mask[ti] == 1:
                    i, j = truth_i[ti], truth_j[ti]
                    a = best_n[ti]
                    obj_mask[b, a, j, i] = 1
                    tgt_mask[b, a, j, i, :] = 1
                    target[b, a, j, i, 0] = truth_x_all[b, ti] - \
                        truth_x_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 1] = truth_y_all[b, ti] - \
                        truth_y_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 2] = torch.log(
                        truth_w_all[b, ti] / torch.Tensor(self.masked_anchors)[best_n[ti], 0] + 1e-16)
                    target[b, a, j, i, 3] = torch.log(
                        truth_h_all[b, ti] / torch.Tensor(self.masked_anchors)[best_n[ti], 1] + 1e-16)
                    target[b, a, j, i, 4] = 1
                    # target[b, a, j, i, 5 + x_label[b, ti,
                    #                               0].to(torch.int16).numpy()] = 1
                    tgt_scale[b, a, j, i, :] = torch.sqrt(
                        2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize)

        # loss calculation
        # output: [B,3,8,8,5]

        output[..., 4] *= obj_mask
        output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
        output[..., 2:4] *= tgt_scale

        target[..., 4] *= obj_mask
        target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
        target[..., 2:4] *= tgt_scale

        bceloss = nn.BCELoss(weight=tgt_scale*tgt_scale)  # weighted BCEloss
        bceloss_sup = nn.BCELoss(weight=tgt_scale[:output.size(0)//2,...]*tgt_scale[:output.size(0)//2,...])  # weighted BCEloss
        if semi == False:
            loss_xy = bceloss(output[..., :2], target[..., :2])
            loss_wh = self.l2_loss(output[..., 2:4], target[..., 2:4]) / 2
            loss_obj = self.bce_loss(output[..., 4], target[..., 4])
            # loss_l2 = self.l2_loss(output, target)

            # loss_seg=nn.BCEWithLogitsLoss()(mask,y_label)*float(batchsize)
            loss_det = loss_xy + loss_wh + loss_obj
            loss_det*=(float(batchsize))

            loss=loss_det.sum()
            loss_seg=torch.zeros_like(loss)

            return loss,loss_det,loss_seg
        else:
            loss_xy_sup = bceloss_sup(output[:output.size(0)//2,..., :2], target[:output.size(0)//2,..., :2])
            loss_wh_sup = self.l2_loss(output[:output.size(0)//2,..., 2:4], target[:output.size(0)//2,..., 2:4]) / 2
            loss_obj_sup = self.bce_loss(output[:output.size(0)//2,..., 4], target[:output.size(0)//2,..., 4])
            loss_obj_unsup = self.bce_loss(output[output.size(0)//2:,..., 4], target[output.size(0)//2:,..., 4])
            # loss_l2 = self.l2_loss(output, target)

            # loss_seg=nn.BCEWithLogitsLoss()(mask,y_label)*float(batchsize)
            loss_det_sup = loss_xy_sup + loss_wh_sup + loss_obj_sup
            loss_det_unsup = loss_obj_unsup
            loss_det_sup*=(float(batchsize/2))
            loss_det_unsup*=(float(batchsize/2))

            loss_sup=loss_det_sup.sum()
            loss_unsup=loss_det_unsup.sum()
            loss_seg_sup=torch.zeros_like(loss_sup)
            loss_seg_unsup=torch.zeros_like(loss_unsup)

            return loss_sup,loss_det_sup,loss_seg_sup,loss_unsup,loss_det_unsup,loss_seg_unsup