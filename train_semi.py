from utils.distributed import *
import torch.multiprocessing as mp
from utils.ckpt import *
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.logging import *
import argparse
import time
from utils import config
from datasets.dataloader import loader, RefCOCODataSet, RefCOCODataSet_semi
from tensorboardX import SummaryWriter
from utils.utils import *
import torch.optim as Optim
from importlib import import_module
import torch.nn.functional as F
from utils.utils import EMA
from collections import OrderedDict
import torch

def validate_iter(__C,
             net,
             loader,
             writer,
             epoch,
             iter,
             rank,
             ix_to_token,
             save_ids=None,
             prefix='Val',
             ema=None,
             student=True):
    if ema is not None:
        ema.apply_shadow()
    net.eval()

    batches = len(loader)
    batch_time = AverageMeter('Time', ':6.5f')
    data_time = AverageMeter('Data', ':6.5f')
    losses = AverageMeter('Loss', ':.4f')
    box_ap = AverageMeter('BoxIoU@0.5', ':6.2f')
    mask_ap = AverageMeter('MaskIoU', ':6.2f')
    inconsistency_error = AverageMeter('IE', ':6.2f')
    mask_aps={}
    for item in np.arange(0.5, 1, 0.05):
        mask_aps[item]=[]
    meters = [batch_time, data_time, losses, box_ap, mask_ap,inconsistency_error]
    meters_dict = {meter.name: meter for meter in meters}
    progress = ProgressMeter(__C.VERSION, __C.EPOCHS, len(loader), meters, prefix=prefix+': ')
    with th.no_grad():
        end = time.time()
        for ith_batch, data in enumerate(loader):
            ref_iter, image_iter, mask_iter, box_iter,gt_box_iter, mask_id, info_iter = data
            ref_iter = ref_iter.cuda( non_blocking=True)
            image_iter = image_iter.cuda( non_blocking=True)
            box_iter = box_iter.cuda( non_blocking=True)
            box, mask= net(image_iter, ref_iter)


            gt_box_iter=gt_box_iter.squeeze(1)
            gt_box_iter[:, 2] = (gt_box_iter[:, 0] + gt_box_iter[:, 2])
            gt_box_iter[:, 3] = (gt_box_iter[:, 1] + gt_box_iter[:, 3])
            gt_box_iter=gt_box_iter.cpu().numpy()
            info_iter=info_iter.cpu().numpy()
            box=box.squeeze(1).cpu().numpy()
            pred_box_vis=box.copy()

            ###predictions to gt
            for i in range(len(gt_box_iter)):
                box[i]=yolobox2label(box[i],info_iter[i])


            box_iou=batch_box_iou(torch.from_numpy(gt_box_iter),torch.from_numpy(box)).cpu().numpy()
            seg_iou=[]
            mask=mask.cpu().numpy()
            for i, mask_pred in enumerate(mask):
                if writer is not None and save_ids is not None and ith_batch*__C.BATCH_SIZE+i in save_ids:
                    ixs=ref_iter[i].cpu().numpy()
                    words=[]
                    for ix in ixs:
                        if ix >0:
                            words.append(ix_to_token[ix])
                    sent=' '.join(words)
                    box_iter = box_iter.view(box_iter.shape[0], -1) * __C.INPUT_SHAPE[0]
                    box_iter[:, 0] = box_iter[:, 0] - 0.5 * box_iter[:, 2]
                    box_iter[:, 1] = box_iter[:, 1] - 0.5 * box_iter[:, 3]
                    box_iter[:, 2] = box_iter[:, 0] + box_iter[:, 2]
                    box_iter[:, 3] = box_iter[:, 1] + box_iter[:, 3]
                    det_image=draw_visualization(normed2original(image_iter[i],__C.MEAN,__C.STD),sent,pred_box_vis[i].cpu().numpy(),box_iter[i].cpu().numpy())
                    writer.add_image('image/' + str(ith_batch * __C.BATCH_SIZE + i) + '_det',det_image,epoch,dataformats='HWC')
                    writer.add_image('image/' + str(ith_batch * __C.BATCH_SIZE + i) + '_seg', (mask[i,None]*255).astype(np.uint8))


                mask_gt=np.load(os.path.join(__C.MASK_PATH[__C.DATASET],'%d.npy'%mask_id[i]))
                mask_pred=mask_processing(mask_pred,info_iter[i])

                single_seg_iou,single_seg_ap=mask_iou(mask_gt,mask_pred)
                for item in np.arange(0.5, 1, 0.05):
                    mask_aps[item].append(single_seg_ap[item]*100.)
                seg_iou.append(single_seg_iou)
            seg_iou=np.array(seg_iou).astype(np.float32)

            ie=(box_iou>=0.5).astype(np.float32)*(seg_iou<0.5).astype(np.float32)+(box_iou<0.5).astype(np.float32)*(seg_iou>=0.5).astype(np.float32)
            inconsistency_error.update(ie.mean()*100., ie.shape[0])
            box_ap.update((box_iou>0.5).astype(np.float32).mean()*100., box_iou.shape[0])
            mask_ap.update(seg_iou.mean()*100., seg_iou.shape[0])

            reduce_meters(meters_dict, rank, __C)

            if (ith_batch % __C.PRINT_FREQ == 0 or ith_batch==(len(loader)-1)) and main_process(__C,rank):
                progress.display(epoch, ith_batch)
            batch_time.update(time.time() - end)
            end = time.time()

        if main_process(__C,rank) and writer is not None:
            writer.add_scalar("Acc/BoxIoU@0.5_100", box_ap.avg_reduce, global_step=iter)
            if student == True:
                writer.add_scalar("Acc/BoxIoU@0.5_student_100", box_ap.avg_reduce, global_step=iter)
            else:
                writer.add_scalar("Acc/BoxIoU@0.5_teacher_100", box_ap.avg_reduce, global_step=iter)
    if ema is not None:
        ema.restore()
    return box_ap.avg_reduce, mask_ap.avg_reduce

@torch.no_grad()
def _update_teacher_model(student,teacher,word_size,keep_rate=0.9996):
    student_model_dict = student.state_dict()
    difference=0.
    new_teacher_dict = OrderedDict()
    for key, value in teacher.state_dict().items():
        if key in student_model_dict.keys():
            new_teacher_dict[key] = (
                    student_model_dict[key] * (1 - keep_rate) + value * keep_rate
            )
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
                    loader_label,
                    loader_unlabel,
                    val_loader,
                    ix_to_token,
                    scalar,
                    writer,
                    epoch,
                    rank,
                    best_det_acc_teacher,
                    best_det_acc_student,
                    best_teacher_step,
                    best_student_step,
                    ema=None):
    student.train()
    if __C.MULTIPROCESSING_DISTRIBUTED:
        loader_label.sampler.set_epoch(epoch)
        loader_unlabel.sampler.set_epoch(epoch)
    loader_label.set_length(max(len(loader_label),len(loader_unlabel)))
    loader_unlabel.set_length(max(len(loader_label),len(loader_unlabel)))
    loader = enumerate(zip(loader_label,loader_unlabel))

    batches = len(loader_label)
    nb=len(loader_label)
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
    progress = ProgressMeter(__C.VERSION,__C.EPOCHS, batches, meters, prefix='Train: ')
    end = time.time()
    
    for ith_batch, data in loader:
        ni=ith_batch+epoch*nb
        data_time.update(time.time() - end)


        (ref_iter,image_iter,mask_iter,box_iter,gt_box_iter,mask_id,info_iter), (
            ref_iter_unlabel,image_iter_unlabel,mask_iter_unlabel,box_iter_unlabel,gt_box_iter_unlabel,mask_id_unlabel,info_iter_unlabel) = data
        ref_iter = ref_iter.cuda(non_blocking=True)
        image_iter = image_iter.cuda(non_blocking=True)
        mask_iter = mask_iter.cuda(non_blocking=True)
        box_iter = box_iter.cuda(non_blocking=True)
        ref_iter_unlabel = ref_iter_unlabel.cuda(non_blocking=True)
        image_iter_unlabel = image_iter_unlabel.cuda(non_blocking=True)

        loss_semi, loss_det_semi, loss_seg_semi=torch.zeros(1).cuda(),torch.zeros(1).cuda(),torch.zeros(1).cuda()

        #random resize
        if len(__C.MULTI_SCALE)>1:
            h,w=__C.MULTI_SCALE[np.random.randint(0,len(__C.MULTI_SCALE))]
            image_iter=F.interpolate(image_iter,(h,w))
            mask_iter=F.interpolate(mask_iter,(h,w))
            image_iter_unlabel_resize=F.interpolate(image_iter_unlabel,(h,w))
        else:
            image_iter_unlabel_resize=image_iter_unlabel.clone()

        student.train()
        if ni < __C.BURN_UP:
            if scalar is not None:
                with th.cuda.amp.autocast():
                    loss_sup, loss_det_sup, loss_seg_sup = student(image_iter,ref_iter,det_label=box_iter,seg_label=mask_iter)
            else:
                loss_sup, loss_det_sup, loss_seg_sup = student(image_iter, ref_iter, det_label=box_iter,seg_label=mask_iter)
            loss = loss_sup

        elif ni >= __C.BURN_UP and (
                ni - __C.BURN_UP
        ) % __C.SEMI_UPDATE_ITER == 0:
            if ni == __C.BURN_UP and (not __C.STAC):
                _update_teacher_model(student, teacher, word_size=len(__C.GPU), keep_rate=0.)
                print("Going to semi-supervised stage...")

            teacher.eval()
            with torch.no_grad():
                pseudo_box, pseudo_mask = teacher(image_iter_unlabel, ref_iter_unlabel)
                info_iter_unlabel=info_iter_unlabel.cpu().numpy()
                pseudo_box=pseudo_box.squeeze(1).cpu().numpy()
                ###predictions to gt
                for i in range(len(gt_box_iter_unlabel)):
                    pseudo_box[i]=yolobox2label(pseudo_box[i],info_iter_unlabel[i])
                pseudo_box[:, 2] = (pseudo_box[:, 2]-pseudo_box[:, 0])
                pseudo_box[:, 3] = (pseudo_box[:, 3]-pseudo_box[:, 1])
                from datasets.dataloader import label2yolobox
                sized_pseudo_box = np.zeros((len(gt_box_iter_unlabel),5))
                for i in range(len(gt_box_iter_unlabel)):
                    sized_pseudo_box[i]=label2yolobox(pseudo_box[i].reshape(1,-1),tuple(info_iter_unlabel[i]),__C.INPUT_SHAPE[0],lrflip=__C.FLIP_LR)
                sized_pseudo_box = torch.from_numpy(sized_pseudo_box[:, :4]).cuda(non_blocking=True)
                pseudo_mask = pseudo_mask.unsqueeze(1)
                sized_pseudo_box = sized_pseudo_box.unsqueeze(1)

            if len(__C.MULTI_SCALE)>1:
                pseudo_mask=F.interpolate(pseudo_mask,(h,w))
            image_iter = torch.cat([image_iter,image_iter_unlabel_resize.clone()],0).cuda(non_blocking=True)
            ref_iter = torch.cat([ref_iter, ref_iter_unlabel.clone()],0).cuda(non_blocking=True)
            box_iter=torch.cat([box_iter,sized_pseudo_box],0).cuda(non_blocking=True)
            mask_iter=torch.cat([mask_iter,pseudo_mask],0).cuda(non_blocking=True)
                

            if scalar is not None:
                with th.cuda.amp.autocast():
                    loss_sup,loss_det_sup,loss_seg_sup,loss_semi,loss_det_semi,loss_seg_semi = student(image_iter,ref_iter,det_label=box_iter,seg_label=mask_iter,semi=True)
            else:
                loss_sup,loss_det_sup,loss_seg_sup,loss_semi,loss_det_semi,loss_seg_semi = student(image_iter,ref_iter,det_label=box_iter,seg_label=mask_iter,semi=True)

            loss=loss_sup+loss_semi*__C.SEMI_LOSS_WEIGHT

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

            if __C.GRAD_NORM_CLIP > 0:
                nn.utils.clip_grad_norm_(
                    student.parameters(),
                    __C.GRAD_NORM_CLIP
                )
            optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update_params()
        
        if ni>=__C.BURN_UP and (not __C.STAC):
            _update_teacher_model(student, teacher, word_size=len(__C.GPU), keep_rate=__C.SEMI_EMA)

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
            writer.add_scalar("lr/train", optimizer.param_groups[0]["lr"], global_step=global_step)
            if ith_batch % __C.PRINT_FREQ == 0 or ith_batch==batches:
                progress.display(epoch, ith_batch)
        # break
        batch_time.update(time.time() - end)
        
        step = epoch * batches + ith_batch
        if step % __C.VALIDATE_FREQ == 0 and step != 0:
            box_ap_student,mask_ap=validate_iter(__C,student,val_loader,writer,epoch,step,rank,ix_to_token,ema=ema,student=True)
        if step % __C.VALIDATE_FREQ == 0 and step != 0 and step >= __C.BURN_UP:
            box_ap_teacher,mask_ap=validate_iter(__C,teacher,val_loader,writer,epoch,step,rank,ix_to_token,ema=None,student=False)

        if main_process(__C,rank) and step % __C.VALIDATE_FREQ == 0 and step != 0:
            if step >= __C.BURN_UP:
                if box_ap_teacher>best_det_acc_teacher:
                    best_det_acc_teacher=box_ap_teacher
                    best_teacher_step = step
                    torch.save({'epoch': epoch + 1, 'state_dict': teacher.state_dict(), 'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict(),'lr':optimizer.param_groups[0]["lr"],},
                                os.path.join(__C.LOG_PATH, str(__C.VERSION),'ckpt', 'det_best_teacher.pth'))
                    print('Save teacher checkpoint...')
                print('Teacher best acc: ', best_det_acc_teacher)
                print('Teacher best step: ', best_teacher_step)
            if ema is not None:
                ema.apply_shadow()
            if step >= __C.VALIDATE_FREQ:
                if box_ap_student>best_det_acc_student:
                    best_det_acc_student=box_ap_student
                    best_student_step = step
                    torch.save({'epoch': epoch + 1, 'state_dict': student.state_dict(), 'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict(),'lr':optimizer.param_groups[0]["lr"],},
                                os.path.join(__C.LOG_PATH, str(__C.VERSION),'ckpt', 'det_best_student.pth'))
                    print('Save student checkpoint...')
                print('Student best acc: ', best_det_acc_student)
                print('Student best step: ', best_student_step)

            if ema is not None:
                    ema.restore()
        end = time.time()
    return best_det_acc_teacher,best_det_acc_student,best_teacher_step,best_student_step
        

def main_worker(gpu,__C):
    global best_det_acc_teacher,best_det_acc_student,best_seg_acc,best_teacher_step,best_student_step
    best_det_acc_teacher,best_det_acc_student,best_seg_acc=0.,0.,0.
    best_teacher_step,best_student_step=0,0
    if __C.MULTIPROCESSING_DISTRIBUTED:
        if __C.DIST_URL == "env://" and __C.RANK == -1:
            __C.RANK = int(os.environ["RANK"])
        if __C.MULTIPROCESSING_DISTRIBUTED:
            __C.RANK = __C.RANK* len(__C.GPU) + gpu
        dist.init_process_group(backend=dist.Backend('NCCL'), init_method=__C.DIST_URL, world_size=__C.WORLD_SIZE, rank=__C.RANK)

    train_set_label=RefCOCODataSet_semi(__C,split='train',sup=False,label=True)
    train_loader_label=loader(__C,train_set_label,gpu,shuffle=(not __C.MULTIPROCESSING_DISTRIBUTED),drop_last=True)

    train_set_unlabel=RefCOCODataSet_semi(__C,split='train',sup=False,label=False)
    train_loader_unlabel=loader(__C,train_set_unlabel,gpu,shuffle=(not __C.MULTIPROCESSING_DISTRIBUTED),drop_last=True)

    val_set=RefCOCODataSet_semi(__C,split='val',sup=False,label=True)
    val_loader=loader(__C,val_set,gpu,shuffle=False)

    teacher= ModelLoader(__C).Net(
        __C,
        train_set_label.pretrained_emb,
        train_set_label.token_size
    )

    student= ModelLoader(__C).Net(
        __C,
        train_set_label.pretrained_emb,
        train_set_label.token_size
    )

    #optimizer
    params = filter(lambda p: p.requires_grad, student.parameters())
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

    scheduler = get_lr_scheduler(__C,optimizer,max(len(train_loader_label),len(train_loader_unlabel)))

    start_epoch = 0

    if os.path.isfile(__C.RESUME_PATH):
        checkpoint = torch.load(__C.RESUME_PATH,map_location=torch.device('cpu')) # lambda storage, loc: storage.cuda()
        new_dict = {}
        for k in checkpoint['state_dict']:
            if 'module.' in k:
                new_k = k.replace('module.', '')
                new_dict[new_k] = checkpoint['state_dict'][k]
        if len(new_dict.keys())==0:
            new_dict=checkpoint['state_dict']
        teacher.load_state_dict(new_dict,strict=False)

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
        best_det_acc_teacher,best_det_acc_student,best_teacher_step,best_student_step = train_one_epoch(__C,teacher,student,optimizer,scheduler,train_loader_label,train_loader_unlabel,val_loader,val_set.ix_to_token,scalar,writer,ith_epoch,gpu,best_det_acc_teacher,best_det_acc_student,best_teacher_step,best_student_step,ema)

    if __C.MULTIPROCESSING_DISTRIBUTED:
        cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description="RealGIN or SimREC")
    parser.add_argument('--config', type=str, required=True, default='./config/sup/realgin_sup_baseline.yaml')
    parser.add_argument('--resume-weights', type=str, required=True, default='')
    args=parser.parse_args()
    assert args.config is not None
    __C = config.load_cfg_from_cfg_file(args.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in __C.GPU)
    setup_unique_version(__C)
    seed_everything(__C.SEED)
    N_GPU=len(__C.GPU)
    __C.RESUME_PATH=args.resume_weights
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
