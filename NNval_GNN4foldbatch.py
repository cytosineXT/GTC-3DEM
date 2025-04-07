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
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from matplotlib.ticker import FuncFormatter
import argparse

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] 
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT)) 
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

def plotRCS2(rcs,savedir,logger):
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio
    tic = time.time()
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
    if cutmax != None:
        plt.clim(cutmin, cutmax)
    plt.xlabel("Theta")
    plt.ylabel("Phi")
    plt.savefig(savedir)
    plt.close()
    if logger!=None:
        # logger.info(f'画图用时：{time.time()-tic:.4f}s')
        1
    else:
        print(f'draw time consume：{time.time()-tic:.4f}s')

def plotstatistic2(psnr_list, ssim_list, mse_list, statisticdir):
    def to_percent(y,position):
        return str(int((100*y)))
    binss = 20

    plt.clf()
    plt.figure(figsize=(12, 6))

    #-----------------------------------mse-------------------------------------------
    print(len(mse_list))
    plt.subplot(3, 3, 1)
    counts, bins, patches = plt.hist(mse_list, bins=binss, edgecolor='black', density=True)

    mu, std = norm.fit(mse_list)
    # x = np.linspace(-5, 15, 1000)
    x = np.linspace(min(mse_list), max(mse_list), 1000)
    plt.plot(x, norm.pdf(x, mu, std), 'r-', linewidth=2, label='Normal fit')
    plt.xlabel('MSE')
    # plt.ylabel('Probability of samples')
    plt.ylabel('Probability of samples (%)')
    plt.title('MSE Histogram and Normal Fit')
    plt.legend()


    #-----------------------------------PSNR-------------------------------------------
    plt.subplot(3, 3, 2)
    counts, bins, patches = plt.hist(psnr_list, bins=binss, edgecolor='black', density=True)
    fomatter=FuncFormatter(to_percent)
    plt.gca().yaxis.set_major_formatter(fomatter)
    mu, std = norm.fit(psnr_list)
    # x = np.linspace(15,45, 1000)
    x = np.linspace(min(psnr_list), max(psnr_list), 1000)
    plt.plot(x, norm.pdf(x, mu, std), 'r-', linewidth=2, label='Normal fit')
    plt.xlabel('PSNR')
    # plt.ylabel('Probability of samples')
    plt.ylabel('Probability of samples (%)')
    plt.title('PSNR Histogram and Normal Fit')
    plt.legend()

    #-----------------------------------SSIM-------------------------------------------
    plt.subplot(3, 3, 3)
    # counts, bins, patches = plt.hist(ssim_list, bins=binss, edgecolor='black', density=True, stacked=True)
    counts, bins, patches = plt.hist(ssim_list, bins=binss, edgecolor='black', density=True)
    mu, std = norm.fit(ssim_list)
    # x = np.linspace(0.6,1.1, 1000)
    x = np.linspace(min(ssim_list), max(ssim_list), 1000)
    plt.plot(x, norm.pdf(x, mu, std), 'r-', linewidth=2, label='Normal fit')
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

    autoencoder = MeshCodec( 
    num_discrete_coors = 128,
    device= device,
    paddingsize = paddingsize,
    attn_encoder_depth = attnlayer,
                            ).to(device)
    autoencoder.load_state_dict(torch.load(weight), strict=True)
    # autoencoder = autoencoder.to(device)
    #-------------------------------------------------------------------------------------
    with torch.no_grad():
        for in_em1,rcs1 in tqdm(dataloader,desc=f'val process:',ncols=70,postfix=f''):
            in_em0 = in_em1.copy()
            objlist , _ = find_matching_files(in_em1[0], "./testplane")
            planesur_faces, planesur_verts, planesur_faceedges, geoinfo = process_files(objlist, device)

            start_time0 = time.time()
            loss, outrcs, psnr_mean, psnrlist, ssim_mean, ssimlist, mse, nmse, rmse, l1, percentage_error, mselist= autoencoder( #这里使用网络，是进去跑了forward
                vertices = planesur_verts,
                faces = planesur_faces, #torch.Size([batchsize, 33564, 3])
                face_edges = planesur_faceedges,
                in_em = in_em1,#.to(device)
                GT = rcs1.to(device), 
                logger = logger,
                device = device,
            )
            inftime = time.time()-start_time0

            if trainval == False:
                logger.info(f'one batch inference：{time.time()-start_time0:.4f}s，average one sample inference：{(time.time()-start_time0)/batchsize:.4f}s')
                # logger.info(f'{plane}, em={eminfo}, loss={loss:.4f}')
            # torch.cuda.empty_cache()
            if draw == True:
                for i in range(outrcs.shape[0]): 
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
                    plot2DRCS(rcs=single_outrcs, savedir=out2Drcspngpath, logger=logger,cutmax=None)
                    plot2DRCS(rcs=single_outrcs, savedir=out2Drcspngpath2, logger=logger,cutmax=torch.max(single_rcs1).item())
                    plot2DRCS(rcs=single_diff, savedir=out2Drcspngpath3, logger=logger,cutmax=0.05,cutmin=-0.05)
                    plot2DRCS(rcs=single_rcs1, savedir=out2DGTpngpath, logger=logger,cutmax=None)

                    if draw3d == True:
                        plotRCS2(rcs=single_rcs1, savedir=outGTpngpath, logger=logger)
                        plotRCS2(rcs=single_outrcs, savedir=outrcspngpath, logger=logger)
            
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
            logger.info(f"using model weight{weight} test {len(losses)} samples, Mean Loss: {ave_loss:.4f}, Mean PSNR: {ave_psnr:.2f}dB, Mean SSIM: {ave_ssim:.4f}, Mean MSE:{ave_mse:.4f}")
            logger.info(f'test set dir:{rcsdir}, total time consume:{time.strftime("%H:%M:%S", time.gmtime(time.time()-tic))}')
            logger.info(f"damaged files：{corrupted_files}")
        logger.info(f'val set:{rcsdir}, total time consume:{time.strftime("%H:%M:%S", time.gmtime(time.time()-tic))}')
        logger.info(f'↑----val loss:{ave_loss:.4f},psnr:{ave_psnr:.2f},ssim:{ave_ssim:.4f},mse:{ave_mse:.4f},inftime:{ave_inftime:.4f}s----↑')

        statisdir = os.path.join(save_dir,f'sta/statistic_epoch{epoch}_PSNR{ave_psnr:.2f}dB_SSIM{ave_ssim:.4f}_MSE:{ave_mse:.4f}_Loss{ave_loss:.4f}.png')
        if not os.path.exists(os.path.dirname(statisdir)):
            os.makedirs(os.path.dirname(statisdir))
        plotstatistic2(psnrs,ssims,mses,statisdir)
        savefigdata(psnrs,img_path=os.path.join(save_dir,f'sta/valall_epoch{epoch}psnrs{ave_psnr:.2f}.png'))
        savefigdata(ssims,img_path=os.path.join(save_dir,f'sta/valall_epoch{epoch}ssims{ave_ssim:.4f}.png'))
        savefigdata(mses,img_path=os.path.join(save_dir,f'sta/valall_epoch{epoch}mses{ave_mse:.4f}.png'))
    return ave_mse, ave_psnr, ave_ssim, psnrs, ssims, mses  #ave_psnr, 

def parse_args():
    parser = argparse.ArgumentParser(description="Script with customizable parameters using argparse.")
    parser.add_argument('--valdir', type=str, default='testrcs', help='Path to validation directory')
    parser.add_argument('--pretrainweight', type=str, default=None, help='Path to pretrained weights')
    parser.add_argument('--attn', type=int, default=1, help='Transformer layers')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    weight = args.pretrainweight
    valdir = args.valdir
    attnlayer = args.attn
    
    trainval = False
    # cuda = 'cuda:0'
    cuda = 'cpu'
    draw = True
    # draw = False
    draw3d = False
    lgrcs = False
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    batchsize = 40
    
    from datetime import datetime
    date = datetime.today().strftime("%m%d")
    save_dir = str(increment_path(Path(ROOT / "output" / "inference" /f'{date}_{valdir}'), exist_ok=False))
    logdir = os.path.join(save_dir,'alog.txt')
    logger = get_logger(logdir)
    epoch = -1
    
    valfilelist = os.listdir(valdir)
    valdataset = EMRCSDataset(valfilelist, valdir)
    valdataloader = DataLoader.DataLoader(valdataset, batch_size=batchsize, shuffle=False, num_workers=16, pin_memory=True)
    if trainval == False:
        logger.info(f'using model weight{weight} inference test set{valdir} and draw')
    valmain(draw, device, weight, valdir, save_dir, logger, epoch, trainval, draw3d, attnlayer=attnlayer, valdataloader=valdataloader, batchsize=batchsize)