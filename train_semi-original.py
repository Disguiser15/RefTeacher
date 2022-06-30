from utils.distributed import *
import torch.multiprocessing as mp
from utils.ckpt import *
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.logging import *
import argparse
import time
from utils import config
from datasets.dataloader import loader,RefCOCODataSet
from tensorboardX import SummaryWriter
from utils.utils import *
import torch.optim as Optim
from importlib import import_module
import torch.nn.functional as F
from test import validate
from utils.utils import  EMA
from collections import OrderedDict
import torch

@torch.no_grad()
def _update_teacher_model(student,teacher,word_size,keep_rate=0.9996):
    student_model_dict = student.state_dict()
    # if word_size > 1:
    #     student_model_dict = {
    #         key[7:]: value for key, value in student.state_dict().items()
    #     }
    # else:
    #     student_model_dict = student.state_dict()
    # print(student_model_dict)
    difference=0.
    new_teacher_dict = OrderedDict()
    for key, value in teacher.state_dict().items():
        if key in student_model_dict.keys():
            new_teacher_dict[key] = (
                    student_model_dict[key] * (1 - keep_rate) + value * keep_rate
            )
            # difference+=(student_model_dict[key]-value).sum()
        else:
            raise Exception("{} is not found in student model".format(key))
    teacher.load_state_dict(new_teacher_dict)

class ModelLoader:
    def __init__(self, __C):

        self.model_use = __C.MODEL
        model_moudle_path = 'models.' + self.model_use + '.net'
        self.model_moudle = import_module(model_moudle_path)

    def Net(self, __arg1, __arg2, __arg3):
        return self.model_moudle.Net(__arg1, __arg2, __arg3)

def train_one_epoch(__C,
                    teacher,
                    student,
                    optimizer,
                    scheduler,
                    loader,
                    scalar,
                    writer,
                    epoch,
                    rank,
                    ema=None):
    student.train()
    if __C.MULTIPROCESSING_DISTRIBUTED:
        loader.sampler.set_epoch(epoch)

    batches = len(loader)
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    losses_sup = AverageMeter('LossS', ':.2f')
    losses_det_sup = AverageMeter('LossDetS', ':.2f')
    losses_seg_sup = AverageMeter('LossSegS', ':.2f')
    losses_semi = AverageMeter('LossU', ':.2f')
    losses_det_semi = AverageMeter('LossDetU', ':.2f')
    losses_seg_semi = AverageMeter('LossSegU', ':.2f')
    lr = AverageMeter('lr', ':.5f')
    meters = [batch_time, data_time, losses_sup,losses_det_sup,losses_seg_sup,
              losses_semi,losses_det_semi,losses_seg_semi,lr]
    meters_dict = {meter.name: meter for meter in meters}
    progress = ProgressMeter(__C.VERSION,__C.EPOCHS, len(loader), meters, prefix='Train: ')
    end = time.time()
    nb=len(loader)
    for ith_batch, data in enumerate(loader):
        ni=ith_batch+epoch*nb
        data_time.update(time.time() - end)

        # ref_iter,image_iter,image_lab_iter,mask_iter,bitmask_full,box_iter,gt_box_iter,mask_id,info_iter= data
        # ref_iter = ref_iter.cuda(non_blocking=True)
        # image_iter = image_iter.cuda(non_blocking=True)
        # image_lab_iter= image_lab_iter.cuda(non_blocking=True)
        # mask_iter = mask_iter.cuda(non_blocking=True)
        # box_iter = box_iter.cuda( non_blocking=True)
        # bitmask_full= bitmask_full.cuda( non_blocking=True)

        ref_iter,image_iter,mask_iter,box_iter,gt_box_iter,mask_id,info_iter= data
        ref_iter = ref_iter.cuda(non_blocking=True)
        image_iter = image_iter.cuda(non_blocking=True)
        mask_iter = mask_iter.cuda(non_blocking=True)
        box_iter = box_iter.cuda(non_blocking=True)

        loss_semi, loss_det_semi, loss_seg_semi=torch.zeros(1).cuda(),torch.zeros(1).cuda(),torch.zeros(1).cuda()

        #random resize
        if len(__C.MULTI_SCALE)>1:
            h,w=__C.MULTI_SCALE[np.random.randint(0,len(__C.MULTI_SCALE))]
            image_iter=F.interpolate(image_iter,(h,w))
            mask_iter=F.interpolate(mask_iter,(h,w))

        # _update_teacher_model(student, teacher, word_size=len(__C.GPU), keep_rate=__C.SEMI_EMA)

        # if ni < __C.BURN_UP:
        #     if scalar is not None:
        #         with th.cuda.amp.autocast():
        #             loss_sup, loss_det_sup, loss_seg_sup = student(image_iter, ref_iter, det_label=box_iter,
        #                                                            seg_label=mask_iter, image_lab_iter=image_lab_iter,bitmask_full=bitmask_full)
        #     else:
        #         loss_sup, loss_det_sup, loss_seg_sup = student(image_iter, ref_iter, det_label=box_iter,
        #                                                        seg_label=mask_iter, image_lab_iter=image_lab_iter,bitmask_full=bitmask_full)

        if ni < __C.BURN_UP:
            if scalar is not None:
                with th.cuda.amp.autocast():
                    loss_sup, loss_det_sup, loss_seg_sup = student(image_iter,ref_iter,det_label=box_iter,seg_label=mask_iter)
            else:
                loss_sup, loss_det_sup, loss_seg_sup = student(image_iter, ref_iter, det_label=box_iter,seg_label=mask_iter)
            loss = loss_sup

        elif ni>=__C.BURN_UP and (
                ni -  __C.BURN_UP
        ) % __C.SEMI_UPDATE_ITER == 0:
            if ni==__C.BURN_UP:
                _update_teacher_model(student, teacher, word_size=len(__C.GPU), keep_rate=0.)
                print("Going to semi-supervised stage...")
            # else:
            #     _update_teacher_model(student, teacher, word_size=len(__C.GPU), keep_rate=__C.SEMI_EMA)
            teacher.eval()
            with torch.no_grad():
                pseudo_box, pseudo_mask = teacher(image_iter, ref_iter)
                info_iter=info_iter.cpu().numpy()
                pseudo_box=pseudo_box.squeeze(1).cpu().numpy()
                ###predictions to gt
                for i in range(len(gt_box_iter)):
                    pseudo_box[i]=yolobox2label(pseudo_box[i],info_iter[i])
                pseudo_box[:, 2] = (pseudo_box[:, 2]-pseudo_box[:, 0])
                pseudo_box[:, 3] = (pseudo_box[:, 3]-pseudo_box[:, 1])
                from datasets.dataloader import label2yolobox
                sized_pseudo_box = np.zeros((len(gt_box_iter),5))
                for i in range(len(gt_box_iter)):
                    sized_pseudo_box[i]=label2yolobox(pseudo_box[i].reshape(1,-1),tuple(info_iter[i]),__C.INPUT_SHAPE[0],lrflip=__C.FLIP_LR)
                sized_pseudo_box = torch.from_numpy(sized_pseudo_box[:, :4]).cuda(non_blocking=True)
                pseudo_mask = pseudo_mask.unsqueeze(1)
                sized_pseudo_box = sized_pseudo_box.unsqueeze(1)

                # pseudo_box = pseudo_box[:, :4]
                # pseudo_box[:, 0] = (pseudo_box[:, 0] + pseudo_box[:, 2]) / 2./image_iter.shape[2]
                # pseudo_box[:, 1] = (pseudo_box[:, 1] + pseudo_box[:, 3]) / 2./image_iter.shape[3]
                # pseudo_box[:, 2] = (pseudo_box[:, 2] - pseudo_box[:, 0]) * 2./image_iter.shape[2]
                # pseudo_box[:, 3] = (pseudo_box[:, 3] - pseudo_box[:, 1]) * 2./image_iter.shape[3]
                # pseudo_mask = pseudo_mask.unsqueeze(1)
                # pseudo_box = pseudo_box.unsqueeze(1)
            image_iter = torch.cat([image_iter,image_iter.clone()],0)
            ref_iter = torch.cat([ref_iter, ref_iter.clone()], 0)
            # image_lab_iter = torch.cat([image_lab_iter, image_lab_iter.clone()], 0)
            # bitmask_full=torch.cat([bitmask_full,bitmask_full.clone()],0)
            box_iter=torch.cat([box_iter,sized_pseudo_box],0)
            mask_iter=torch.cat([mask_iter,pseudo_mask],0)
            # if scalar is not None:
            #     with th.cuda.amp.autocast():
            #         loss_sup, loss_det_sup, loss_seg_sup,loss_seg_semi = student(image_iter, ref_iter, det_label=box_iter,
            #                                                        seg_label=mask_iter, image_lab_iter=image_lab_iter,bitmask_full=bitmask_full,semi=True)
            # else:
            #     loss_sup, loss_det_sup, loss_seg_sup,loss_seg_semi = student(image_iter, ref_iter, det_label=box_iter,
            #                                                    seg_label=mask_iter, image_lab_iter=image_lab_iter,bitmask_full=bitmask_full,semi=True)
            if scalar is not None:
                with th.cuda.amp.autocast():
                    loss_sup,loss_det_sup,loss_seg_sup,loss_semi,loss_det_semi,loss_seg_semi = student(image_iter,ref_iter,det_label=box_iter,seg_label=mask_iter,semi=True)
            else:
                loss_sup,loss_det_sup,loss_seg_sup,loss_semi,loss_det_semi,loss_seg_semi = student(image_iter,ref_iter,det_label=box_iter,seg_label=mask_iter,semi=True)

            loss=loss_sup+loss_semi*__C.SEMI_LOSS_WEIGHT
            # loss=loss_sup
        optimizer.zero_grad()
        if scalar is not None:
            scalar.scale(loss).backward()
            scalar.step(optimizer)
            if __C.GRAD_NORM_CLIP > 0:
                nn.utils.clip_grad_norm_(
                    student.parameters(),
                    __C.GRAD_NORM_CLIP
                )
            scalar.update()
        else:
            loss.backward(retain_graph=True)
            # for p in list(filter(lambda p: p.grad is not None, net.parameters())):
            #     print(p.grad.data)
            if __C.GRAD_NORM_CLIP > 0:
                nn.utils.clip_grad_norm_(
                    student.parameters(),
                    __C.GRAD_NORM_CLIP
                )
            optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update_params()
        
        _update_teacher_model(student, teacher, word_size=len(__C.GPU), keep_rate=__C.SEMI_EMA)
        # _update_teacher_model(student, teacher, word_size=len(__C.GPU), keep_rate=0.)

        losses_sup.update(loss_sup.item(), image_iter.size(0))
        losses_det_sup.update(loss_det_sup.item(), image_iter.size(0))
        losses_seg_sup.update(loss_seg_sup.item(), image_iter.size(0))
        losses_semi.update(loss_semi.item(), image_iter.size(0))
        losses_det_semi.update(loss_det_semi.item(), image_iter.size(0))
        losses_seg_semi.update(loss_seg_semi.item(), image_iter.size(0))
        lr.update(optimizer.param_groups[0]["lr"],-1)
        reduce_meters(meters_dict, rank, __C)
        if main_process(__C,rank):
            global_step = epoch * batches + ith_batch
            writer.add_scalar("loss_sup/train", losses_sup.avg_reduce, global_step=global_step)
            writer.add_scalar("loss_det_sup/train", losses_det_sup.avg_reduce, global_step=global_step)
            writer.add_scalar("loss_seg_sup/train", losses_seg_sup.avg_reduce, global_step=global_step)
            writer.add_scalar("loss_semi/train", losses_semi.avg_reduce, global_step=global_step)
            writer.add_scalar("loss_det_semi/train", losses_det_semi.avg_reduce, global_step=global_step)
            writer.add_scalar("loss_seg_semi/train", losses_seg_semi.avg_reduce, global_step=global_step)
            if ith_batch % __C.PRINT_FREQ == 0 or ith_batch==len(loader):
                progress.display(epoch, ith_batch)
        # break
        batch_time.update(time.time() - end)
        end = time.time()


def main_worker(gpu,__C):
    global best_det_acc,best_seg_acc
    best_det_acc,best_seg_acc=0.,0.
    if __C.MULTIPROCESSING_DISTRIBUTED:
        if __C.DIST_URL == "env://" and __C.RANK == -1:
            __C.RANK = int(os.environ["RANK"])
        if __C.MULTIPROCESSING_DISTRIBUTED:
            __C.RANK = __C.RANK* len(__C.GPU) + gpu
        # print(__C.RANK)
        #
        dist.init_process_group(backend=dist.Backend('NCCL'), init_method=__C.DIST_URL, world_size=__C.WORLD_SIZE, rank=__C.RANK)

    train_set=RefCOCODataSet(__C,split='train')
    train_loader=loader(__C,train_set,gpu,shuffle=(not __C.MULTIPROCESSING_DISTRIBUTED),drop_last=True)

    val_set=RefCOCODataSet(__C,split='val')
    val_loader=loader(__C,val_set,gpu,shuffle=False)

    teacher= ModelLoader(__C).Net(
        __C,
        train_set.pretrained_emb,
        train_set.token_size
    )

    student= ModelLoader(__C).Net(
        __C,
        train_set.pretrained_emb,
        train_set.token_size
    )

    #optimizer
    params = filter(lambda p: p.requires_grad, student.parameters())#split_weights(net)
    std_optim = getattr(Optim, __C.OPT)

    eval_str = 'params, lr=%f'%__C.LR
    for key in __C.OPT_PARAMS:
        eval_str += ' ,' + key + '=' + str(__C.OPT_PARAMS[key])
    optimizer=eval('std_optim' + '(' + eval_str + ')')

    ema=None


    if __C.MULTIPROCESSING_DISTRIBUTED:
        torch.cuda.set_device(gpu)
        student = DDP(student.cuda(), device_ids=[gpu],find_unused_parameters=True)
        teacher = DDP(teacher.cuda(), device_ids=[gpu],find_unused_parameters=True)
    elif len(gpu)==1:
        teacher.cuda()
        student.cuda()
    else:
        teacher = DP(teacher.cuda())
        student = DP(student.cuda())


    if main_process(__C, gpu):
        print(__C)
        print(student)
        total = sum([param.nelement() for param in student.parameters()])
        print('  + Number of all params: %.2fM' % (total / 1e6))  # 每一百万为一个单位
        total = sum([param.nelement() for param in student.parameters() if param.requires_grad])
        print('  + Number of trainable params: %.2fM' % (total / 1e6))  # 每一百万为一个单位

    scheduler = get_lr_scheduler(__C,optimizer,len(train_loader))

    start_epoch = 0

    if os.path.isfile(__C.RESUME_PATH):
        checkpoint = torch.load(__C.RESUME_PATH,map_location=lambda storage, loc: storage.cuda() )
        new_dict = {}
        for k in checkpoint['state_dict']:
            if 'module.' in k:
                new_k = k.replace('module.', '')
                new_dict[new_k] = checkpoint['state_dict'][k]
        if len(new_dict.keys())==0:
            new_dict=checkpoint['state_dict']
        student.load_state_dict(new_dict,strict=False)
        student.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        if main_process(__C,gpu):
            print("==> loaded checkpoint from {}\n".format(__C.RESUME_PATH) +
                  "==> epoch: {} lr: {} ".format(checkpoint['epoch'],checkpoint['lr']))

    if os.path.isfile(__C.VL_PRETRAIN_WEIGHT):
        checkpoint = torch.load(__C.VL_PRETRAIN_WEIGHT,map_location=lambda storage, loc: storage.cuda() )
        new_dict = {}
        for k in checkpoint['state_dict']:
            if 'module.' in k:
                new_k = k.replace('module.', '')
                new_dict[new_k] = checkpoint['state_dict'][k]
        if len(new_dict.keys())==0:
            new_dict=checkpoint['state_dict']
        student.load_state_dict(new_dict,strict=False)
        start_epoch = 0
        if main_process(__C,gpu):
            print("==> loaded checkpoint from {}\n".format(__C.VL_PRETRAIN_WEIGHT) +
                  "==> epoch: {} lr: {} ".format(checkpoint['epoch'],checkpoint['lr']))



    if __C.AMP:
        assert th.__version__ >= '1.6.0', \
            "Automatic Mixed Precision training only supported in PyTorch-1.6.0 or higher"
        scalar = th.cuda.amp.GradScaler()
    else:
        scalar = None

    if main_process(__C,gpu):
        writer = SummaryWriter(log_dir=os.path.join(__C.LOG_PATH,str(__C.VERSION)))
    else:
        writer = None

    save_ids=np.random.randint(1, len(val_loader) * __C.BATCH_SIZE, 100) if __C.LOG_IMAGE else None

    for ith_epoch in range(start_epoch, __C.EPOCHS):
        if __C.USE_EMA and ema is None:
            ema = EMA(student, 0.9997)
        train_one_epoch(__C,teacher,student,optimizer,scheduler,train_loader,scalar,writer,ith_epoch,gpu,ema)
        box_ap,mask_ap=validate(__C,student,val_loader,writer,ith_epoch,gpu,val_set.ix_to_token,save_ids=save_ids,ema=ema)
        box_ap,mask_ap=validate(__C,teacher,val_loader,writer,ith_epoch,gpu,val_set.ix_to_token,save_ids=save_ids,ema=ema)
        if main_process(__C,gpu):
            if ema is not None:
                ema.apply_shadow()
            torch.save({'epoch': ith_epoch + 1, 'state_dict': teacher.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),'lr':optimizer.param_groups[0]["lr"],},
                       os.path.join(__C.LOG_PATH, str(__C.VERSION),'ckpt', 'last.pth'))
            if box_ap>best_det_acc:
                best_det_acc=box_ap
                torch.save({'epoch': ith_epoch + 1, 'state_dict': teacher.state_dict(), 'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),'lr':optimizer.param_groups[0]["lr"],},
                           os.path.join(__C.LOG_PATH, str(__C.VERSION),'ckpt', 'det_best.pth'))
            if mask_ap>best_seg_acc:
                best_seg_acc=mask_ap
                torch.save({'epoch': ith_epoch + 1, 'state_dict': teacher.state_dict(), 'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),'lr':optimizer.param_groups[0]["lr"],},
                           os.path.join(__C.LOG_PATH, str(__C.VERSION),'ckpt', 'seg_best.pth'))
            if ema is not None:
                ema.restore()
    if __C.MULTIPROCESSING_DISTRIBUTED:
        cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description="SimREC")
    # parser.add_argument('--config', type=str, default='./config/SimREC_RefCOCO_scratch_semi.yaml')
    parser.add_argument('--config', type=str, required=True, default='./config/refcoco.yaml')
    args=parser.parse_args()
    assert args.config is not None
    __C = config.load_cfg_from_cfg_file(args.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in __C.GPU)
    setup_unique_version(__C)
    seed_everything(__C.SEED)
    N_GPU=len(__C.GPU)

    if not os.path.exists(os.path.join(__C.LOG_PATH,str(__C.VERSION))):
        os.makedirs(os.path.join(__C.LOG_PATH,str(__C.VERSION),'ckpt'),exist_ok=True)

    if N_GPU == 1:
        __C.MULTIPROCESSING_DISTRIBUTED = False
    else:
        # turn on single or multi node multi gpus training
        __C.MULTIPROCESSING_DISTRIBUTED = True
        __C.WORLD_SIZE *= N_GPU
        __C.DIST_URL = f"tcp://127.0.0.1:{find_free_port()}"
    if __C.MULTIPROCESSING_DISTRIBUTED:
        mp.spawn(main_worker, args=(__C,), nprocs=N_GPU, join=True)
    else:
        main_worker(__C.GPU,__C)


if __name__ == '__main__':
    main()
