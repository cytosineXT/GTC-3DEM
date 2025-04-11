import torch
import time
from tqdm import tqdm
# from net.jxtnet_GNNn0115 import MeshCodec
from net.jxtnet_GNNn0118acEn import MeshCodec
import torch.utils.data.dataloader as DataLoader
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')
from pathlib import Path
from net.utils_newload import increment_path, EMRCSDataset, MultiEMRCSDataset, get_logger, get_model_memory, psnr, ssim, find_matching_files, process_files, WrappedModel, savefigdata
from NNval_GNN4foldbatch import  plot2DRCS, valmain, plotstatistic2
from pytictoc import TicToc
t = TicToc()
t.tic()
import random
import numpy as np
import argparse
from thop import profile
import copy

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     torch.backends.cudnn.benchmark = False 
     torch.backends.cudnn.deterministic = True
     np.random.seed(seed)
     random.seed(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Script with customizable parameters using argparse.")
    parser.add_argument('--epoch', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=4, help='batchsize')
    parser.add_argument('--valbatch', type=int, default=40, help='valbatchsize')
    parser.add_argument('--use_preweight', type=bool, default=False, help='Whether to use pretrained weights')
    parser.add_argument('--smooth', type=bool, default=False, help='Whether to use pretrained weights')
    parser.add_argument('--draw', type=bool, default=True, help='Whether to enable drawing')

    parser.add_argument('--trainname', type=str, default='heldout_test', help='logname')
    parser.add_argument('--folder', type=str, default='testtrain', help='exp output folder name')
    parser.add_argument('--mode', type=str, default='fasttest', help='10train 50fine 100fine fasttest')
    parser.add_argument('--loss', type=str, default='L1', help='L1 best, mse 2nd')
    parser.add_argument('--rcsdir', type=str, default='testrcs', help='Path to rcs directory')
    parser.add_argument('--valdir', type=str, default='testrcs', help='Path to validation directory')
    parser.add_argument('--pretrainweight', type=str, default=None, help='Path to pretrained weights')

    parser.add_argument('--seed', type=int, default=7, help='Random seed for reproducibility')
    parser.add_argument('--attn', type=int, default=1, help='Transformer layers')
    parser.add_argument('--gama', type=float, default=0.001, help='control max loss, i love 0.001')
    parser.add_argument('--beta', type=float, default=0., help='seems to be control contrastive loss, i forgot, useless, 0')
    parser.add_argument('--lr', type=float, default=0.001, help='Loss threshold or gamma parameter')
    parser.add_argument('--cuda', type=str, default='cuda:0', help='CUDA device to use(cpu cuda:0 cuda:1...)')
    parser.add_argument('--fold', type=str, default=None, help='Fold to use for validation (None fold1 fold2 fold3 fold4)')
    return parser.parse_args()

tic0 = time.time()
tic = time.time()
print('code start time:',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))  

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] 
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT)) 
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

args = parse_args()

epoch = args.epoch
use_preweight = args.use_preweight
smooth = args.smooth
draw = args.draw
pretrainweight = args.pretrainweight
seed = args.seed
attnlayer = args.attn
gama = args.gama
beta = args.beta
learning_rate = args.lr
cudadevice = args.cuda
name = args.trainname
folder = args.folder
mode = args.mode
batchsize = args.batch
valbatch = args.valbatch
loss_type = args.loss

datafolder = '/mnt/SrvDataDisk/Datasets_3DEM/allplanes/mie'

Fold1 = ['b871','bb7d','b827','b905','bbc6']
Fold2 = ['b80b','ba0f','b7c1','b9e6','bb7c']
Fold3 = ['b943','b97b','b812','bc2c','b974']
Fold4 = ['bb26','b7fd','baa9','b979','b8ed']

if args.fold: 
    fold_mapping = {
        'fold1': Fold1,
        'fold2': Fold2,
        'fold3': Fold3,
        'fold4': Fold4,
    }
    val_planes = fold_mapping[args.fold]
    train_planes = [files for fold in [Fold1, Fold2, Fold3, Fold4] if fold != val_planes for files in fold]
    valdir = None
    rcsdir = None

else: 
    rcsdir = args.rcsdir
    valdir = args.valdir

# setup_seed(seed)
if args.seed is not None:
    seed = args.seed
    setup_seed(args.seed)
    print(f"use provided seed: {args.seed}")
else:
    random_seed = torch.randint(0, 10000, (1,)).item()
    setup_seed(random_seed)
    print(f"not provide seed, use random seed: {random_seed}")
    seed = random_seed

accumulation_step = 8
threshold = 20
bestloss = 1
epoch_mean_loss = 0.0
minmse = 1.0
valmse = 1.0
in_ems = []
rcss = []
cnt = 0
losses = [] 
psnrs = []
ssims = []
mses = []
nmses, rmses, l1s, percentage_errors = [], [], [], []
corrupted_files = []
lgrcs = False
shuffle = True
multigpu = False
alpha = 0.0
lr_time = epoch

encoder_layer = 6
decoder_outdim = 12  # 3S 6M 12L
paddingsize = 18000

from datetime import datetime
date = datetime.today().strftime("%m%d")
save_dir = str(increment_path(Path(ROOT / "output" / f"{folder}" / f'{date}_sd{seed}_{mode}{loss_type}_{args.fold if args.fold else ""}{name}_e{epoch}Tr{attnlayer}_{cudadevice}_'), exist_ok=False))

lastsavedir = os.path.join(save_dir,'last.pt')
bestsavedir = os.path.join(save_dir,'best.pt')
maxsavedir = os.path.join(save_dir,'minmse.pt')
lossessavedir = os.path.join(save_dir,'loss.png')
psnrsavedir = os.path.join(save_dir,'psnr.png')
ssimsavedir = os.path.join(save_dir,'ssim.png')
msesavedir = os.path.join(save_dir,'mse.png')
nmsesavedir = os.path.join(save_dir,'nmse.png')
rmsesavedir = os.path.join(save_dir,'rmse.png')
l1savedir = os.path.join(save_dir,'l1.png')
valmsesavedir = os.path.join(save_dir,'valmse.png')
valpsnrsavedir = os.path.join(save_dir,'valpsnr.png')
valssimsavedir = os.path.join(save_dir,'valssim.png')
valmsesavedir2 = os.path.join(save_dir,'Trainvalmse.png')
valpsnrsavedir2 = os.path.join(save_dir,'Trainvalpsnr.png')
valssimsavedir2 = os.path.join(save_dir,'Trainvalssim.png')
percentage_errorsavedir = os.path.join(save_dir,'percentage_error.png')
allinonesavedir = os.path.join(save_dir,'allinone.png')
logdir = os.path.join(save_dir,'log.txt')
logger = get_logger(logdir)
logger.info(args)
logger.info(f'seed:{seed}')

if args.fold:
    logger.info(f'dataset setting:{args.fold} ,val on {val_planes}, train on {train_planes}, mode={mode}')
    val_mse_per_plane = {plane: [] for plane in val_planes}
    val_psnr_per_plane = {plane: [] for plane in val_planes}
    val_ssim_per_plane = {plane: [] for plane in val_planes}
    valallpsnrs = []
    valallssims = []
    valallmses = []
    allavemses = []
    allavepsnrs = []
    allavessims = []
    
    if mode=='10train' or 'fasttest': #10train 50fine 100fine
        train_files = [plane + '_mie_10train' for plane in train_planes]
    elif mode=='50fine':
        train_files = [plane + '_mie_50train' for plane in train_planes]
    elif mode=='100fine':
        train_files = [plane + '_mie_train' for plane in train_planes]
    
    val_files = [plane + '_mie_val' for plane in val_planes]

    dataset = MultiEMRCSDataset(train_files, datafolder)
    dataloader = DataLoader.DataLoader(dataset, batch_size=batchsize, shuffle=shuffle, num_workers=16, pin_memory=True)
    val_dataloaders = {} 
    for valfile1 in val_files:
        valdataset = MultiEMRCSDataset([valfile1], datafolder)
        plane1 = valfile1[:4]
        val_dataloaders[plane1] = DataLoader.DataLoader(valdataset, batch_size=valbatch, shuffle=False, num_workers=16, pin_memory=True)

    logger.info(f'train set samples:{dataset.__len__()}，single val set samples:{valdataset.__len__()}，val set count:{len(val_dataloaders)}，tatal val set samples:{valdataset.__len__()*len(val_dataloaders)}')

else:
    logger.info(f'train set is{rcsdir}')
    filelist = os.listdir(rcsdir)
    dataset = EMRCSDataset(filelist, rcsdir)
    dataloader = DataLoader.DataLoader(dataset, batch_size=batchsize, shuffle=shuffle, num_workers=16, pin_memory=True) #这里调用的是getitem

    valfilelist = os.listdir(valdir)
    valdataset = EMRCSDataset(valfilelist, valdir) #这里进的是init
    valdataloader = DataLoader.DataLoader(valdataset, batch_size=valbatch, shuffle=shuffle, num_workers=16, pin_memory=True) #transformer的话40才行？20.。 纯GNN的话60都可以
    logger.info(f'train set samples:{dataset.__len__()}，val set samples:{valdataset.__len__()}')

logger.info(f'saved to {lastsavedir}')

device = torch.device(cudadevice if torch.cuda.is_available() else "cpu")
# device = 'cpu'
logger.info(f'device:{device}')

autoencoder = MeshCodec(
    num_discrete_coors = 128,
    device= device,
    paddingsize = paddingsize,
    attn_encoder_depth = attnlayer,
)
get_model_memory(autoencoder,logger)
total_params = sum(p.numel() for p in autoencoder.parameters())
logger.info(f"Total parameters: {total_params}")

if use_preweight == True:
    autoencoder.load_state_dict(torch.load(pretrainweight), strict=True)
    logger.info(f'successfully load pretrain_weight:{pretrainweight}')
else:
    logger.info('not use pretrain_weight, starting new train')

autoencoder = autoencoder.to(device)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=lr_time)

flag = 1
GTflag = 1
flopflag = 1
for i in range(epoch):
    psnr_list, ssim_list, mse_list, nmse_list, rmse_list, l1_list, percentage_error_list = [], [], [], [], [], [], []
    jj=0
    logger.info('\n')
    epoch_loss = []
    timeepoch = time.time()
    for in_em1,rcs1 in tqdm(dataloader,desc=f'epoch:{i+1},train prosecc,lr={scheduler.get_last_lr()[0]:.5f}',ncols=130,postfix=f'former epoch:{i},loss_mean:{(epoch_mean_loss):.4f}'):

        jj=jj+1
        in_em0 = in_em1.copy()
        # optimizer.zero_grad()
        objlist , ptlist = find_matching_files(in_em1[0], "./testplane")
        # objlist , ptlist = find_matching_files(in_em1[0], "./planes")
        planesur_faces, planesur_verts, planesur_faceedges, geoinfo = process_files(objlist, device)

        loss, outrcs, psnr_mean, _, ssim_mean, _, mse, nmse, rmse, l1, percentage_error, _ = autoencoder(
            vertices = planesur_verts,
            faces = planesur_faces, #torch.Size([batchsize, 33564, 3])
            face_edges = planesur_faceedges,
            # geoinfo = geoinfo, #[area, volume, scale]
            in_em = in_em1,#.to(device)
            GT = rcs1.to(device),
            logger = logger,
            device = device,
            lgrcs = lgrcs,
            gama=gama,
            beta=beta,
            loss_type=loss_type,
            smooth=smooth
        )
        if flopflag == 1:
            temp_model = copy.deepcopy(autoencoder)
            wrapped_model = WrappedModel(temp_model)
            flops, params = profile(wrapped_model, (planesur_verts, planesur_faces, planesur_faceedges, in_em1, rcs1.to(device),device))
            logger.info(f' params:{params / 1000000.0:.2f}M, Gflops:{flops / 1000000000.0:.2f}G')
            flopflag = 0
            del temp_model 

        if lgrcs == True:
            outrcslg = outrcs
            outrcs = torch.pow(10, outrcs)
        if batchsize > 1:
            lossback=loss.mean() / accumulation_step 
            lossback.backward() 
        else:
            outem = [int(in_em1[1]), int(in_em1[2]), float(f'{in_em1[3].item():.3f}')]
            tqdm.write(f'em:{outem},loss:{loss.item():.4f}')
            lossback=loss / accumulation_step
            lossback.backward()
        epoch_loss.append(loss.item())

        torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=threshold)
        if (jj) % accumulation_step == 0 or (jj) == len(dataloader):
            optimizer.step() 
            optimizer.zero_grad()
        psnr_list.append(psnr_mean)
        ssim_list.append(ssim_mean)
        mse_list.append(mse)
        nmse_list.append(nmse)
        rmse_list.append(rmse)
        l1_list.append(l1)
        percentage_error_list.append(percentage_error)
        

        in_em0[1:] = [tensor.to(device) for tensor in in_em0[1:]]
        if flag == 1:
            drawrcs = outrcs[0].unsqueeze(0)
            drawem = torch.stack(in_em0[1:]).t()[0]
            drawGT = rcs1[0][:-1,:].unsqueeze(0)
            drawplane = in_em0[0][0]
            flag = 0
        for j in range(torch.stack(in_em0[1:]).t().shape[0]):
            if flag == 0 and torch.equal(torch.stack(in_em0[1:]).t()[j], drawem):
                drawrcs = outrcs[j].unsqueeze(0)
                break
    logger.info(save_dir)

    p = psnr(drawrcs.to(device), drawGT.to(device))
    s = ssim(drawrcs.to(device), drawGT.to(device))
    m = torch.nn.functional.mse_loss(drawrcs.to(device), drawGT.to(device))
    if GTflag == 1:
        outGTpngpath = os.path.join(save_dir,f'{drawplane}theta{drawem[0]}phi{drawem[1]}freq{drawem[2]}_GT.png')
        out2DGTpngpath = os.path.join(save_dir,f'{drawplane}theta{drawem[0]}phi{drawem[1]}freq{drawem[2]}_2DGT.png')
        plot2DRCS(rcs=drawGT.squeeze(), savedir=out2DGTpngpath, logger=logger,cutmax=None)
        GTflag = 0
        logger.info('drawed GT map')
    if i == 0 or (i+1) % 20 == 0: 
        outrcspngpath = os.path.join(save_dir,f'{drawplane}theta{drawem[0]}phi{drawem[1]}freq{drawem[2]}_epoch{i}.png')
        out2Drcspngpath = os.path.join(save_dir,f'{drawplane}theta{drawem[0]}phi{drawem[1]}freq{drawem[2]}_epoch{i}_psnr{p.item():.2f}_ssim{s.item():.4f}_mse{m:.4f}_2D.png')
        plot2DRCS(rcs=drawrcs.squeeze(), savedir=out2Drcspngpath, logger=logger,cutmax=None)
        logger.info(f'drawed {i+1} epoch map')

    epoch_mean_loss = sum(epoch_loss)/len(epoch_loss)
    losses.append(epoch_mean_loss)
    epoch_psnr = sum(psnr_list)/len(psnr_list) 
    epoch_ssim = sum(ssim_list)/len(ssim_list)
    epoch_mse = sum(mse_list)/len(mse_list)
    epoch_nmse = sum(nmse_list)/len(nmse_list)
    epoch_rmse = sum(rmse_list)/len(rmse_list)
    epoch_l1 = sum(l1_list)/len(l1_list)
    epoch_percentage_error = sum(percentage_error_list)/len(percentage_error_list)
    psnrs.append(epoch_psnr)
    ssims.append(epoch_ssim)
    mses.append(epoch_mse)
    nmses.append(epoch_nmse)
    rmses.append(epoch_rmse)
    l1s.append(epoch_l1)
    percentage_errors.append(epoch_percentage_error)
    logger.info('epoch metrics computed')

    if bestloss > epoch_mean_loss:
        bestloss = epoch_mean_loss
        if os.path.exists(bestsavedir):
            os.remove(bestsavedir)
        torch.save(autoencoder.to('cpu').state_dict(), bestsavedir)
    if os.path.exists(lastsavedir):
        os.remove(lastsavedir)
    torch.save(autoencoder.to('cpu').state_dict(), lastsavedir)
    logger.info('model weight saved')
    autoencoder.to(device)

    scheduler.step()
    logger.info('lr scheduled')

    logger.info(f'↓-----------------------this epoch time consume：{time.strftime("%H:%M:%S", time.gmtime(time.time()-timeepoch))}-----------------------↓')
    logger.info(f'↑----epoch:{i+1}(lr:{scheduler.get_last_lr()[0]:.4f}),loss:{epoch_mean_loss:.4f},psnr:{epoch_psnr:.2f},ssim:{epoch_ssim:.4f},mse:{epoch_mse:.4f}----↑')
    
    plt.clf()
    plt.figure(figsize=(7, 4.5))
    plt.plot(range(0, i+1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.savefig(lossessavedir)
    savefigdata(losses,img_path=lossessavedir)
    plt.close()
    
    plt.clf()
    plt.plot(range(0, i+1), psnrs)
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.title('Training PSNR Curve')
    plt.savefig(psnrsavedir)
    savefigdata(psnrs,img_path=psnrsavedir)
    plt.close()

    plt.clf()
    plt.plot(range(0, i+1), ssims)
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('Training SSIM Curve')
    plt.savefig(ssimsavedir)
    savefigdata(ssims,img_path=ssimsavedir)
    plt.close()

    plt.clf()
    plt.plot(range(0, i+1), mses)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Training MSE Curve')
    plt.savefig(msesavedir)
    savefigdata(mses,img_path=msesavedir)
    plt.close()

    plt.clf()
    plt.plot(range(0, i+1), nmses)
    plt.xlabel('Epoch')
    plt.ylabel('NMSE')
    plt.title('Training NMSE Curve')
    plt.savefig(nmsesavedir)
    savefigdata(nmses,img_path=nmsesavedir)
    plt.close()

    plt.clf()
    plt.plot(range(0, i+1), rmses)
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Training RMSE Curve')
    plt.savefig(rmsesavedir)
    plt.close()
    savefigdata(rmses,img_path=rmsesavedir)


    plt.clf()
    plt.plot(range(0, i+1), l1s)
    plt.xlabel('Epoch')
    plt.ylabel('L1')
    plt.title('Training L1 Curve')
    plt.savefig(l1savedir)
    plt.close()
    savefigdata(l1s,img_path=l1savedir)

    plt.clf()
    plt.plot(range(0, i+1), percentage_errors)
    plt.xlabel('Epoch')
    plt.ylabel('Percentage Error')
    plt.title('Training Percentage Error Curve')
    plt.savefig(percentage_errorsavedir)
    plt.close()
    savefigdata(percentage_errors,img_path=percentage_errorsavedir)

    plt.clf() 
    plt.plot(range(0, i+1), losses, label='Loss', color='black')
    plt.plot(range(0, i+1), mses, label='MSE', color='blue')
    plt.plot(range(0, i+1), rmses, label='RMSE', color='green')
    plt.plot(range(0, i+1), l1s, label='L1', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Training Error Curves')
    plt.legend()
    plt.savefig(allinonesavedir)
    plt.close()

    if args.fold:
        for plane, valdataloader in val_dataloaders.items():
            logger.info(f"val on aircraft{plane}")
            valplanedir=os.path.join(save_dir,plane)
            if not os.path.exists(valplanedir):
                os.makedirs(valplanedir)
            if mode == "10train":
                if (i+1) % 1 == 0 or i == -1: 
                    if (i+1) % 100 == 0 or i+1==epoch: 
                        valmse, valpsnr, valssim, valpsnrs, valssims, valmses =valmain(draw=True, device=device, weight=lastsavedir, rcsdir=valdir, save_dir=valplanedir, logger=logger, epoch=i, trainval=True, draw3d=False, lgrcs=lgrcs, decoder_outdim=decoder_outdim,encoder_layer=encoder_layer,paddingsize=paddingsize,valdataloader=valdataloader, attnlayer=attnlayer, batchsize=valbatch)
                    else:
                        valmse, valpsnr, valssim, valpsnrs, valssims, valmses =valmain(draw=False, device=device, weight=lastsavedir, rcsdir=valdir, save_dir=valplanedir, logger=logger, epoch=i, trainval=True, draw3d=False, lgrcs=lgrcs, decoder_outdim=decoder_outdim,encoder_layer=encoder_layer,paddingsize=paddingsize,valdataloader=valdataloader, attnlayer=attnlayer, batchsize=valbatch)
            elif mode == "fasttest":
                if (i+1) % 1 == 0 or i == -1: 
                    if i+1==epoch:
                        valmse, valpsnr, valssim, valpsnrs, valssims, valmses =valmain(draw=True, device=device, weight=lastsavedir, rcsdir=valdir, save_dir=valplanedir, logger=logger, epoch=i, trainval=True, draw3d=False, lgrcs=lgrcs, decoder_outdim=decoder_outdim,encoder_layer=encoder_layer,paddingsize=paddingsize,valdataloader=valdataloader, attnlayer=attnlayer, batchsize=valbatch)
                    else:
                        valmse, valpsnr, valssim, valpsnrs, valssims, valmses =valmain(draw=False, device=device, weight=lastsavedir, rcsdir=valdir, save_dir=valplanedir, logger=logger, epoch=i, trainval=True, draw3d=False, lgrcs=lgrcs, decoder_outdim=decoder_outdim,encoder_layer=encoder_layer,paddingsize=paddingsize,valdataloader=valdataloader, attnlayer=attnlayer, batchsize=valbatch)
            else :
                if (i+1) % 1 == 0 or i == -1:
                    if (i+1) % 2 == 0 or i+1==epoch:
                        valmse, valpsnr, valssim, valpsnrs, valssims, valmses =valmain(draw=True, device=device, weight=lastsavedir, rcsdir=valdir, save_dir=valplanedir, logger=logger, epoch=i, trainval=True, draw3d=False, lgrcs=lgrcs, decoder_outdim=decoder_outdim,encoder_layer=encoder_layer,paddingsize=paddingsize,valdataloader=valdataloader, attnlayer=attnlayer, batchsize=valbatch)
                    else:
                        valmse, valpsnr, valssim, valpsnrs, valssims, valmses =valmain(draw=False, device=device, weight=lastsavedir, rcsdir=valdir, save_dir=valplanedir, logger=logger, epoch=i, trainval=True, draw3d=False, lgrcs=lgrcs, decoder_outdim=decoder_outdim,encoder_layer=encoder_layer,paddingsize=paddingsize,valdataloader=valdataloader, attnlayer=attnlayer, batchsize=valbatch)
            val_mse_per_plane[plane].append(valmse.item())
            val_psnr_per_plane[plane].append(valpsnr.item())
            val_ssim_per_plane[plane].append(valssim.item())

            valallpsnrs.extend(valpsnrs) 
            valallssims.extend(valssims)
            valallmses.extend(valmses) 
        ave_psnr = sum(valallpsnrs)/len(valallpsnrs)
        ave_ssim = sum(valallssims)/len(valallssims)
        ave_mse = sum(valallmses)/len(valallmses)
        allavemses.append(ave_mse)
        allavepsnrs.append(ave_psnr)
        allavessims.append(ave_ssim)

        statisdir = os.path.join(save_dir,f'sta/statisticAll_epoch{i}_PSNR{ave_psnr:.2f}dB_SSIM{ave_ssim:.4f}_MSE:{ave_mse:.4f}.png')
        if not os.path.exists(os.path.dirname(statisdir)):
            os.makedirs(os.path.dirname(statisdir))
        plotstatistic2(valallpsnrs,valallssims,valallmses,statisdir)
        savefigdata(valallpsnrs,img_path=os.path.join(save_dir,f'sta/valall_epoch{i}psnrs{ave_psnr:.2f}.png'))
        savefigdata(valallssims,img_path=os.path.join(save_dir,f'sta/valall_epoch{i}ssims{ave_ssim:.4f}.png'))
        savefigdata(valallmses,img_path=os.path.join(save_dir,f'sta/valall_epoch{i}mses{ave_mse:.4f}.png'))
        valmse = ave_mse

        plt.clf()
        for plane, mse_values in val_mse_per_plane.items():
            plt.plot(range(0, i+1), mse_values, label=plane)
            savefigdata(mse_values,img_path=os.path.join(save_dir,f'{plane}_valmse.png'))
        plt.plot(range(0, i+1),allavemses, label='ave', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('Val MSE Curve')
        plt.legend()
        plt.savefig(valmsesavedir)
        plt.close()
        savefigdata(allavemses,img_path=valmsesavedir)

        plt.clf()
        for plane, psnr_values in val_psnr_per_plane.items():
            plt.plot(range(0, i+1), psnr_values, label=plane)
            savefigdata(psnr_values,img_path=os.path.join(save_dir,f'{plane}_valpsnr.png'))
        plt.plot(range(0, i+1),allavepsnrs, label='ave', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('PSNR')
        plt.title('Val PSNR Curve')
        plt.legend()
        plt.savefig(valpsnrsavedir)
        plt.close()
        savefigdata(allavepsnrs,img_path=valpsnrsavedir)


        plt.clf()
        for plane, ssim_values in val_ssim_per_plane.items():
            plt.plot(range(0, i+1), ssim_values, label=plane)
            savefigdata(ssim_values,img_path=os.path.join(save_dir,f'{plane}_valssim.png'))
        plt.plot(range(0, i+1),allavessims, label='ave', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('SSIM')
        plt.title('Val SSIM Curve')
        plt.legend()
        plt.savefig(valssimsavedir)
        plt.close()
        savefigdata(allavessims,img_path=valssimsavedir)


        lastmse = {k: v[-1] for k, v in val_mse_per_plane.items() if v}
        lastpsnr = {k: v[-1] for k, v in val_psnr_per_plane.items() if v}
        lastssim = {k: v[-1] for k, v in val_ssim_per_plane.items() if v}
        logger.info(f'epoch{i} every aircraft val mse:{lastmse},\npsnr:{lastpsnr},\nssim:{lastssim}')
        logger.info(f'total average val mse:{ave_mse:.4f},psnr:{ave_psnr:.2f},ssim:{ave_ssim:.4f}')

        plt.clf()
        for plane, mse_values in val_mse_per_plane.items():
            plt.plot(range(0, i+1), mse_values, label=plane)
        plt.plot(range(0, i+1),allavemses, label='val ave', linestyle='--')
        plt.plot(range(0, i+1),mses, label='train ave', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('Val MSE Curve')
        plt.legend()
        plt.savefig(valmsesavedir2)
        plt.close()

        plt.clf()
        for plane, psnr_values in val_psnr_per_plane.items():
            plt.plot(range(0, i+1), psnr_values, label=plane)
        plt.plot(range(0, i+1),allavepsnrs, label='val ave', linestyle='--')
        plt.plot(range(0, i+1),psnrs, label='train ave', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('PSNR')
        plt.title('Val PSNR Curve')
        plt.legend()
        plt.savefig(valpsnrsavedir2)
        plt.close()

        plt.clf()
        for plane, ssim_values in val_ssim_per_plane.items():
            plt.plot(range(0, i+1), ssim_values, label=plane)
        plt.plot(range(0, i+1),allavessims, label='val ave', linestyle='--')
        plt.plot(range(0, i+1),ssims, label='train ave', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('SSIM')
        plt.title('Val SSIM Curve')
        plt.legend()
        plt.savefig(valssimsavedir2)
        plt.close()

    else:
        if mode == "10train":
            if (i+1) % 1 == 0 or i == -1: 
                logger.info('every epoch val，every 100 epoch draw')
                if (i+1) % 100 == 0:
                    valmse, valpsnr, valssim, valpsnrs, valssims, valmses =valmain(draw=True, device=device, weight=lastsavedir, rcsdir=valdir, save_dir=save_dir, logger=logger, epoch=i, trainval=True, draw3d=False, lgrcs=lgrcs, decoder_outdim=decoder_outdim,encoder_layer=encoder_layer,paddingsize=paddingsize,valdataloader=valdataloader, attnlayer=attnlayer)
                else:
                    valmse, valpsnr, valssim, valpsnrs, valssims, valmses =valmain(draw=False, device=device, weight=lastsavedir, rcsdir=valdir, save_dir=save_dir, logger=logger, epoch=i, trainval=True, draw3d=False, lgrcs=lgrcs, decoder_outdim=decoder_outdim,encoder_layer=encoder_layer,paddingsize=paddingsize,valdataloader=valdataloader, attnlayer=attnlayer)
                
        elif mode == "fasttest":
            if (i+1) % 1 == 0 or i == -1: 
                logger.info('every epoch val，last epoch draw')
                if i+1==epoch:
                    valmse, valpsnr, valssim, valpsnrs, valssims, valmses =valmain(draw=True, device=device, weight=lastsavedir, rcsdir=valdir, save_dir=save_dir, logger=logger, epoch=i, trainval=True, draw3d=False, lgrcs=lgrcs, decoder_outdim=decoder_outdim,encoder_layer=encoder_layer,paddingsize=paddingsize,valdataloader=valdataloader, attnlayer=attnlayer)
                else:
                    valmse, valpsnr, valssim, valpsnrs, valssims, valmses =valmain(draw=False, device=device, weight=lastsavedir, rcsdir=valdir, save_dir=save_dir, logger=logger, epoch=i, trainval=True, draw3d=False, lgrcs=lgrcs, decoder_outdim=decoder_outdim,encoder_layer=encoder_layer,paddingsize=paddingsize,valdataloader=valdataloader, attnlayer=attnlayer)
        else :
            if (i+1) % 1 == 0 or i == -1:
                logger.info('every epoch val，every 2 epoch draw')
                if (i+1) % 2 == 0 or i+1==epoch:
                    valmse, valpsnr, valssim, valpsnrs, valssims, valmses =valmain(draw=True, device=device, weight=lastsavedir, rcsdir=valdir, save_dir=save_dir, logger=logger, epoch=i, trainval=True, draw3d=False, lgrcs=lgrcs, decoder_outdim=decoder_outdim,encoder_layer=encoder_layer,paddingsize=paddingsize,valdataloader=valdataloader, attnlayer=attnlayer)
                else:
                    valmse, valpsnr, valssim, valpsnrs, valssims, valmses =valmain(draw=False, device=device, weight=lastsavedir, rcsdir=valdir, save_dir=save_dir, logger=logger, epoch=i, trainval=True, draw3d=False, lgrcs=lgrcs, decoder_outdim=decoder_outdim,encoder_layer=encoder_layer,paddingsize=paddingsize,valdataloader=valdataloader, attnlayer=attnlayer)

    if minmse > valmse:
        minmse = valmse
        if os.path.exists(maxsavedir):
            os.remove(maxsavedir)
        torch.save(autoencoder.state_dict(), maxsavedir)

if i+1==epoch:
    renamedir = save_dir+'m'+f'{minmse:.4f}'[2:]
    os.rename(save_dir,renamedir)

logger.info(f"damaged files：{corrupted_files}")
logger.info(f'train finished time：{time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))}')
logger.info(f'train time consume：{(time.time()-tic0)/3600:.2f}小时')
