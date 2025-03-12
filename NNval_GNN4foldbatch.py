import torch
import time
# from net.jxtnet_GNNn0115cond import MeshCodec
from net.jxtnet_GNNn0118acEn import MeshCodec
from net.utils_newload import increment_path, EMRCSDataset, get_logger, find_matching_files, process_files,savefigdata
import torch.utils.data.dataloader as DataLoader
# import trimesh
from pathlib import Path
import sys
import os
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from matplotlib.ticker import FuncFormatter

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def plotRCS2(rcs,savedir,logger):
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio
    tic = time.time()
    # rcs = torch.load('/mnt/Disk/jiangxiaotian/datasets/RCS_mapsmall/RCSmap_theta90phi330f0.9.pt')[:,:,0]
    # print(rcs.shape)
    rcs_np = rcs.detach().cpu().numpy()
    npmax = np.max(rcs_np)
    npmin = np.min(rcs_np)
    theta = np.linspace(0, 2 * np.pi, rcs_np.shape[1])
    phi = np.linspace(0, np.pi, rcs_np.shape[0])
    theta, phi = np.meshgrid(theta, phi)

    x = rcs_np * np.sin(phi) * np.cos(theta)
    y = rcs_np * np.sin(phi) * np.sin(theta)
    z = rcs_np * np.cos(phi)

    fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, cmin = 0, cmax = npmax,  surfacecolor=rcs_np, colorscale='Jet', colorbar=dict(exponentformat='E',title=dict(side='top',text="RCS/m²"), showexponent='all', showticklabels=True, thickness = 30,tick0 = 0, dtick = npmax))])

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
            aspectratio=dict(x=1, y=1, z=0.8),
            aspectmode="manual",
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
        )
    )
    # pio.show(fig)
    pio.write_image(fig, savedir)
    # logger.info(f'画图用时：{time.time()-tic:.4f}s')

def plot2DRCS(rcs, savedir,logger,cutmax,cutmin=None):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize
    if cutmin==None:
        cutmin=torch.min(rcs).item()
    tic = time.time()
    # print(rcs.shape)
    vmin = torch.min(rcs)
    vmax = torch.max(rcs)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.jet

    plt.figure()
    plt.imshow(rcs, cmap=cmap, norm=norm, origin='lower')
    plt.colorbar(label='RCS/m²')
    if cutmax != None:# 设置图例的上下限
        plt.clim(cutmin, cutmax)
    plt.xlabel("Theta")
    plt.ylabel("Phi")
    plt.savefig(savedir)
    plt.close()
    if logger!=None:
        # logger.info(f'画图用时：{time.time()-tic:.4f}s')
        1
    else:
        print(f'画图用时：{time.time()-tic:.4f}s')

def plotstatistic2(psnr_list, ssim_list, mse_list, statisticdir):
    # 绘制统计图
    def to_percent(y,position):
        return str(int((100*y))) #+"%"#这里可以用round（）函数设置取几位小数
    # binss0 = 40
    binss = 20

    plt.clf()
    plt.figure(figsize=(12, 6))

    #-----------------------------------mse-------------------------------------------
    mse_threshold = 0.5
    # mse_list = [m for m in mse_list if m <= mse_threshold]
    print(len(mse_list))
    # MSE 直方图和正态分布曲线
    plt.subplot(3, 3, 1)
    # counts, bins, patches = plt.hist(mse_list, bins=binss, edgecolor='black', density=True, stacked=True)
    counts, bins, patches = plt.hist(mse_list, bins=binss, edgecolor='black', density=True)
    # print(f'counts{counts},bins{bins},patches{patches}')

    # fomatter=FuncFormatter(to_percent)#这里把刻度乘了100，为了得到百分比纵轴
    # plt.gca().yaxis.set_major_formatter(fomatter)

    mu, std = norm.fit(mse_list)
    # x = np.linspace(-5, 15, 1000)
    x = np.linspace(min(mse_list), max(mse_list), 1000)
    plt.plot(x, norm.pdf(x, mu, std), 'r-', linewidth=2, label='Normal fit')
    # plt.xlim(-0.05, max(mse_list)+max(mse_list)/5)  # 限制横坐标范围
    plt.xlabel('MSE')
    # plt.ylabel('Probability of samples')
    plt.ylabel('Probability of samples (%)')
    plt.title('MSE Histogram and Normal Fit')
    plt.legend()


    #-----------------------------------PSNR-------------------------------------------
    # PSNR 直方图和正态分布曲线
    plt.subplot(3, 3, 2)
    # counts, bins, patches = plt.hist(psnr_list, bins=binss, edgecolor='black', density=True, stacked=True)
    counts, bins, patches = plt.hist(psnr_list, bins=binss, edgecolor='black', density=True)
    fomatter=FuncFormatter(to_percent)
    plt.gca().yaxis.set_major_formatter(fomatter)
    mu, std = norm.fit(psnr_list)
    # x = np.linspace(15,45, 1000)
    x = np.linspace(min(psnr_list), max(psnr_list), 1000)
    plt.plot(x, norm.pdf(x, mu, std), 'r-', linewidth=2, label='Normal fit')
    # plt.xlim(-5, 15)  # 限制横坐标范围
    plt.xlabel('PSNR')
    # plt.ylabel('Probability of samples')
    plt.ylabel('Probability of samples (%)')
    plt.title('PSNR Histogram and Normal Fit')
    plt.legend()

    #-----------------------------------SSIM-------------------------------------------
    # SSIM 直方图和正态分布曲线
    plt.subplot(3, 3, 3)
    # counts, bins, patches = plt.hist(ssim_list, bins=binss, edgecolor='black', density=True, stacked=True)
    counts, bins, patches = plt.hist(ssim_list, bins=binss, edgecolor='black', density=True)
    # fomatter=FuncFormatter(to_percent)
    # plt.gca().yaxis.set_major_formatter(fomatter)
    mu, std = norm.fit(ssim_list)
    # x = np.linspace(0.6,1.1, 1000)
    x = np.linspace(min(ssim_list), max(ssim_list), 1000)
    plt.plot(x, norm.pdf(x, mu, std), 'r-', linewidth=2, label='Normal fit')
    # plt.xlim(0.55, 1.1)  # 限制横坐标范围
    plt.xlabel('SSIM')
    # plt.ylabel('Probability of samples')
    plt.ylabel('Probability of samples (%)')
    plt.title('SSIM Histogram and Normal Fit')
    plt.legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig(statisticdir)
    plt.close()


def valmain(draw, device, weight, rcsdir, save_dir, logger, epoch, trainval=False, draw3d=False,lgrcs=False,decoder_outdim=3,encoder_layer=6,paddingsize=18000, n=4, middim=64,attnlayer=0,valdataloader=None, batchsize=40):
    tic = time.time()
    logger.info(f'val batchsize={batchsize}')
    # pngsavedir = os.path.join(save_dir,'0508_b827_theta90phi330freq0.9_4w_sm.png')

    in_ems = []
    rcss = []
    psnrs = []
    ssims = []
    mses = []
    losses = []
    inftimes = []
    corrupted_files = []
    dataloader=valdataloader
    #-------------------------------------------------------------------------------------
    if trainval == False:
        logger.info(f'device:{device}')

    # autoencoder = MeshCodec(num_discrete_coors = 128, paddingsize = paddingsize, attn_encoder_depth=attnlayer).to(device) #这里实例化，是进去跑了init
    autoencoder = MeshCodec( #这里实例化，是进去跑了init 草 但是这里还是用的paddingsize
    num_discrete_coors = 128,
    device= device,
    paddingsize = paddingsize,
    attn_encoder_depth = attnlayer,
                            ).to(device)
    autoencoder.load_state_dict(torch.load(weight), strict=True)
    # autoencoder = autoencoder.to(device)
    #-------------------------------------------------------------------------------------
    with torch.no_grad():
        for in_em1,rcs1 in tqdm(dataloader,desc=f'val进度',ncols=70,postfix=f''):
            in_em0 = in_em1.copy()
            objlist , _ = find_matching_files(in_em1[0], "./testplane")
            planesur_faces, planesur_verts, planesur_faceedges, geoinfo = process_files(objlist, device)

            start_time0 = time.time()
            loss, outrcs, psnr_mean, psnrlist, ssim_mean, ssimlist, mse, nmse, rmse, l1, percentage_error, mselist= autoencoder( #这里使用网络，是进去跑了forward
                vertices = planesur_verts,
                faces = planesur_faces, #torch.Size([batchsize, 33564, 3])
                face_edges = planesur_faceedges,
                in_em = in_em1,#.to(device)
                GT = rcs1.to(device), #这里放真值
                logger = logger,
                device = device,
            )
            inftime = time.time()-start_time0

            if trainval == False:
                logger.info(f'单次推理用时：{time.time()-start_time0:.4f}s，单点推理用时：{(time.time()-start_time0)/batchsize:.4f}s')
                # logger.info(f'{plane}, em={eminfo}, loss={loss:.4f}')
            # torch.cuda.empty_cache()
            if draw == True:
                for i in range(outrcs.shape[0]):  # 遍历每个样本
                    single_outrcs = outrcs[i].squeeze().to(device)
                    single_rcs1 = rcs1[i][:-1].squeeze().to(device)
                    single_diff = single_rcs1-single_outrcs

                    eminfo = [int(in_em0[1][i]), int(in_em0[2][i]), float(in_em0[3][i])]
                    plane = in_em0[0][i]
                    psnr1 = psnrlist[i].item()
                    ssim1 = ssimlist[i].item()
                    mse1 = mselist[i].item()

                    save_dir2 = os.path.join(save_dir,f'epoch{epoch}')
                    Path(save_dir2).mkdir(exist_ok=True)
                    outrcspngpath = os.path.join(save_dir2,f'epoch{epoch}_{plane}_theta{eminfo[0]}phi{eminfo[1]}freq{eminfo[2]:.3f}.png')
                    outGTpngpath = os.path.join(save_dir2,f'epoch{epoch}_{plane}_theta{eminfo[0]}phi{eminfo[1]}freq{eminfo[2]:.3f}_GT.png')

                    out2DGTpngpath = os.path.join(save_dir2,f'epoch{epoch}_{plane}_theta{eminfo[0]}phi{eminfo[1]}freq{eminfo[2]:.3f}_2DGT.png')
                    out2Drcspngpath = os.path.join(save_dir2,f'epoch{epoch}_{plane}_theta{eminfo[0]}phi{eminfo[1]}freq{eminfo[2]:.3f}_psnr{psnr1:.2f}_ssim{ssim1:.4f}_mse{mse1:.4f}_2D.png')
                    out2Drcspngpath2 = os.path.join(save_dir2,f'epoch{epoch}_{plane}_theta{eminfo[0]}phi{eminfo[1]}freq{eminfo[2]:.3f}_psnr{psnr1:.2f}_ssim{ssim1:.4f}_mse{mse1:.4f}_2Dcut.png')
                    out2Drcspngpath3 = os.path.join(save_dir2,f'epoch{epoch}_{plane}_theta{eminfo[0]}phi{eminfo[1]}freq{eminfo[2]:.3f}_psnr{psnr1:.2f}_ssim{ssim1:.4f}_mse{mse1:.4f}_diff{(torch.max(torch.abs(torch.max(single_diff)),torch.abs(torch.min(single_diff)))).item():.4f}_2Ddiff.png')
                    # logger.info(out2Drcspngpath) #查看输出的图片叫啥在哪儿
                    plot2DRCS(rcs=single_outrcs, savedir=out2Drcspngpath, logger=logger,cutmax=None) #预测2D
                    plot2DRCS(rcs=single_outrcs, savedir=out2Drcspngpath2, logger=logger,cutmax=torch.max(single_rcs1).item()) #预测2D但是带cut
                    plot2DRCS(rcs=single_diff, savedir=out2Drcspngpath3, logger=logger,cutmax=0.05,cutmin=-0.05) #预测2D和GT的差异
                    plot2DRCS(rcs=single_rcs1, savedir=out2DGTpngpath, logger=logger,cutmax=None) #GT2D

                    if draw3d == True:
                        plotRCS2(rcs=single_rcs1, savedir=outGTpngpath, logger=logger) #GT 3D
                        plotRCS2(rcs=single_outrcs, savedir=outrcspngpath, logger=logger) #预测 3D
            
            torch.cuda.empty_cache()
            losses.append(loss)
            psnrs.extend(psnrlist)
            ssims.extend(ssimlist)
            mses.extend(mselist)
            inftimes.append(inftime)

        ave_loss = sum(losses)/len(losses)
        ave_psnr = sum(psnrs)/len(psnrs)
        ave_ssim = sum(ssims)/len(ssims)
        ave_mse = sum(mses)/len(mses)
        ave_inftime = sum(inftimes)/len(inftimes)
        if trainval == False:
            logger.info(f"已用{weight}验证{len(losses)}个数据, Mean Loss: {ave_loss:.4f}, Mean PSNR: {ave_psnr:.2f}dB, Mean SSIM: {ave_ssim:.4f}, Mean MSE:{ave_mse:.4f}")
            logger.info(f'val数据集地址:{rcsdir}, 总耗时:{time.strftime("%H:%M:%S", time.gmtime(time.time()-tic))}')
            logger.info(f"损坏的文件：{corrupted_files}")
        logger.info(f'val数据集地址:{rcsdir}, 总耗时:{time.strftime("%H:%M:%S", time.gmtime(time.time()-tic))}')
        logger.info(f'↑----val loss:{ave_loss:.4f},psnr:{ave_psnr:.2f},ssim:{ave_ssim:.4f},mse:{ave_mse:.4f},inftime:{ave_inftime:.4f}s----↑')

        statisdir = os.path.join(save_dir,f'sta/statistic_epoch{epoch}_PSNR{ave_psnr:.2f}dB_SSIM{ave_ssim:.4f}_MSE:{ave_mse:.4f}_Loss{ave_loss:.4f}.png')
        if not os.path.exists(os.path.dirname(statisdir)):
            os.makedirs(os.path.dirname(statisdir))
        plotstatistic2(psnrs,ssims,mses,statisdir)
        savefigdata(psnrs,img_path=os.path.join(save_dir,f'sta/valall_epoch{epoch}psnrs{ave_psnr:.2f}.png'))
        savefigdata(ssims,img_path=os.path.join(save_dir,f'sta/valall_epoch{epoch}ssims{ave_ssim:.4f}.png'))
        savefigdata(mses,img_path=os.path.join(save_dir,f'sta/valall_epoch{epoch}mses{ave_mse:.4f}.png'))
    return ave_mse, ave_psnr, ave_ssim, psnrs, ssims, mses  #ave_psnr, 


if __name__ == '__main__':

    trainval = False
    cuda = 'cuda:0'
    # cuda = 'cpu'
    # draw = True
    draw = False
    draw3d = False
    lgrcs = False
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    batchsize = 40 #纯GNN60 带Transformer40

    # weight = r'/home/jiangxiaotian/workspace/3DEM/outputGNN/train2/0211_sd7_finetuneL1_mul50fold3_GNNTr_e200Tr0_cuda:1_break60/last.pt'
    # weight = r'/home/jiangxiaotian/workspace/3DEM/outputGNN/b7fd_pretrain_m0378.pt'
    weight = r'/home/jiangxiaotian/workspace/3DEM/outputGNN/train0220/0222_sd7_50fineL1_Noneb943_GNNTr_e200Tr1_cuda:1_m0006/last.pt'
    valdir = r'/mnt/truenas_jiangxiaotian/allplanes/mie/b943_mie_val'
    
    from datetime import datetime
    date = datetime.today().strftime("%m%d")
    save_dir = str(increment_path(Path(ROOT / "outputGNN" / "inference" /f'{date}_b943fine50last_sta'), exist_ok=False))
    logdir = os.path.join(save_dir,'alog.txt')
    logger = get_logger(logdir)
    epoch = -1
    attnlayer = 1
    
    valfilelist = os.listdir(valdir)
    valdataset = EMRCSDataset(valfilelist, valdir) #这里进的是init
    valdataloader = DataLoader.DataLoader(valdataset, batch_size=batchsize, shuffle=False, num_workers=16, pin_memory=True)
    if trainval == False:
        logger.info(f'正在用{weight}验证推理{valdir}及画图')
    valmain(draw, device, weight, valdir, save_dir, logger, epoch, trainval, draw3d, attnlayer=attnlayer, valdataloader=valdataloader, batchsize=batchsize)