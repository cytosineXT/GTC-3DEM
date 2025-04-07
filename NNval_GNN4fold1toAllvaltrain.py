import torch
import time
# from net.jxtnet_GNNn0115cond import MeshCodec
from net.jxtnet_GNNn0118acEn import MeshCodec
from net.utils_newload import increment_path, EMRCSDataset, get_logger, find_matching_files, process_files, MultiEMRCSDataset,savefigdata
import torch.utils.data.dataloader as DataLoader
# import trimesh
from pathlib import Path
import sys
import os
from tqdm import tqdm
# import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def polynomial_fit(x, y, degree=3, alpha=0.05):
    """执行多项式拟合并返回拟合曲线和置信区间"""
    # 多项式拟合
    coeffs = np.polyfit(x, y, degree)
    poly = np.poly1d(coeffs)
    
    # 生成拟合曲线
    x_fit = np.linspace(min(x), max(x), 200)
    y_fit = poly(x_fit)
    
    # 计算残差
    residuals = y - poly(x)
    
    # 计算分位数
    lower = np.percentile(residuals, 5)
    upper = np.percentile(residuals, 95)
    
    return x_fit, y_fit, y_fit + lower, y_fit + upper

def plot_metric(ax, x, y, title, ylabel,  color_fit='#D55E00'): #color_point='#4C72B0',
    """通用绘图函数"""
    # 绘制散点
    ax.scatter(x, y, alpha=0.6, edgecolor='white', linewidth=0.5) # color=color_point,
    
    # 执行多项式拟合
    x_fit, y_fit, y_lower, y_upper = polynomial_fit(x, y)
    
    # 绘制拟合曲线
    ax.plot(x_fit, y_fit, color=color_fit, lw=2.5, label='Cubic Fit')
    
    # 绘制置信区间
    ax.fill_between(x_fit, y_lower, y_upper, color='#999999', alpha=0.2, 
                    label='5%-95% Confidence Band')
    
    ax.set_title(title)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.1, 1.0)
    ax.legend()

def plot_scatter_separate(freq_list, mse_list, psnr_list, ssim_list, savedir):
    # 创建保存目录
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    # 设置全局样式
    # plt.style.use('seaborn')
    plt.rcParams.update({'font.size': 12, 'font.family': 'DejaVu Sans'})
    
    # 创建画布
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # # 绘制MSE
    # axes[0].scatter(freq_list, mse_list, alpha=0.6, edgecolor='white', linewidth=0.5)#color='#4C72B0', 
    # upper_5_mse = np.percentile(mse_list, 95)
    # axes[0].axhline(upper_5_mse, color='#CC0000', linestyle='--', 
    #                linewidth=2, label=f'95% Upper Bound\n({upper_5_mse:.4f})')
    # axes[0].set_title('MSE vs Frequency')
    # axes[0].set_xlabel('Frequency (GHz)')
    # axes[0].set_ylabel('MSE')
    # axes[0].grid(True, alpha=0.3)
    # axes[0].set_xlim(0.1, 1.0)
    # axes[0].legend()

    # 绘制PSNR
    plot_metric(axes[0], freq_list, mse_list, 
               'MSE vs Frequency', 'MSE')
    # 绘制PSNR
    plot_metric(axes[1], freq_list, psnr_list, 
               'PSNR vs Frequency', 'PSNR (dB)')
    
    # 绘制SSIM
    plot_metric(axes[2], freq_list, ssim_list, 
               'SSIM vs Frequency', 'SSIM')
    
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, 'combined_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f'可视化结果已保存至: {os.path.abspath(savedir)}')


# def plot_scatter_separate(freq_list, mse_list, psnr_list, ssim_list, savedir, logger):
#     plt.figure(figsize=(10, 6))
#     plt.scatter(freq_list, mse_list, alpha=0.5)
#     plt.xlabel('Frequency (GHz)', fontsize=12)
#     plt.ylabel('MSE', fontsize=12)
#     plt.title('MSE vs Frequency', fontsize=14)
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.xlim(0.1, 1.0)  # 强制设置频率范围
#     plt.savefig(os.path.join(savedir, f'mse_vs_freq.png'), dpi=300, bbox_inches='tight')
#     plt.close()

#     plt.figure(figsize=(10, 6))
#     plt.scatter(freq_list, psnr_list, alpha=0.5)
#     plt.xlabel('Frequency (GHz)', fontsize=12)
#     plt.ylabel('PSNR (dB)', fontsize=12)
#     plt.title('PSNR vs Frequency', fontsize=14)
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.xlim(0.1, 1.0)
#     plt.savefig(os.path.join(savedir, f'psnr_vs_freq.png'), dpi=300, bbox_inches='tight')
#     plt.close()

#     plt.figure(figsize=(10, 6))
#     plt.scatter(freq_list, ssim_list, alpha=0.5)
#     plt.xlabel('Frequency (GHz)', fontsize=12)
#     plt.ylabel('SSIM', fontsize=12)
#     plt.title('SSIM vs Frequency', fontsize=14)
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.xlim(0.1, 1.0)
#     plt.savefig(os.path.join(savedir, f'ssim_vs_freq.png'), dpi=300, bbox_inches='tight')
#     plt.close()
#     logger.info(f'散点图已保存至 {savedir}')

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

def plot2DRCS(rcs, savedir,logger,cutmax):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize
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
        plt.clim(torch.min(rcs).item(), cutmax)
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
        return str(int((100*y)))
    # binss0 = 40
    binss = 20

    plt.clf()
    plt.figure(figsize=(12, 6))

    #-----------------------------------mse-------------------------------------------
    print(len(mse_list))
    plt.subplot(3, 3, 1)
    counts, bins, patches = plt.hist(mse_list, bins=binss*2, edgecolor='black', range=(0,0.75), density=True)

    mu, std = norm.fit(mse_list)
    # x = np.linspace(-5, 15, 1000)
    x = np.linspace(min(mse_list)-0.5, max(mse_list)+0.5, 1000)
    plt.plot(x, norm.pdf(x, mu, std), 'r-', linewidth=2, label='Normal fit')
    plt.xlim(-0.05, max(mse_list)+max(mse_list)/5)  # 限制横坐标范围
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
    x = np.linspace(min(psnr_list)-2, max(psnr_list)+2, 1000)
    plt.plot(x, norm.pdf(x, mu, std), 'r-', linewidth=2, label='Normal fit')
    plt.xlabel('PSNR')
    plt.ylabel('Probability of samples (%)')
    plt.title('PSNR Histogram and Normal Fit')
    plt.legend()

    #-----------------------------------SSIM-------------------------------------------
    # SSIM 直方图和正态分布曲线
    plt.subplot(3, 3, 3)
    counts, bins, patches = plt.hist(ssim_list, bins=binss, edgecolor='black', density=True)
    # plt.gca().yaxis.set_major_formatter(fomatter)
    mu, std = norm.fit(ssim_list)
    x = np.linspace(min(ssim_list)-0.05, max(ssim_list)+0.05, 1000)
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
    freq_list = []
    mse_samples = []
    psnr_samples = []
    ssim_samples = []

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
    #-------------------------------------------------------------------------------------
    with torch.no_grad():
        for in_em1,rcs1 in tqdm(dataloader,desc=f'val进度',ncols=70,postfix=f''):
            in_em0 = in_em1.copy()
            objlist , _ = find_matching_files(in_em1[0], "./planes")
            planesur_faces, planesur_verts, planesur_faceedges, geoinfo = process_files(objlist, device)

            start_time0 = time.time()
            loss, outrcs, psnr_mean, psnrlist, ssim_mean, ssimlist, mse, nmse, rmse, l1, percentage_error, mselist= autoencoder(
                vertices = planesur_verts,
                faces = planesur_faces,
                face_edges = planesur_faceedges,
                in_em = in_em1,
                GT = rcs1.to(device),
                logger = logger,
                device = device,
            )
            inftime = time.time()-start_time0
            for i in range(len(psnrlist)):
                freq = float(in_em0[3][i]) 
                freq_list.append(freq)
                mse_samples.append(mselist[i].item())
                psnr_samples.append(psnrlist[i].item())
                ssim_samples.append(ssimlist[i].item())
                logger.info(f'f={freq:.3f},mse={mselist[i].item():.4f},psnr={psnrlist[i].item():.2f}dB')

            if trainval == False:
                logger.info(f'单次推理用时：{time.time()-start_time0:.4f}s，单点推理用时：{(time.time()-start_time0)/batchsize:.4f}s')
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
                    plot2DRCS(rcs=single_diff, savedir=out2Drcspngpath3, logger=logger,cutmax=torch.max(single_diff).item()) 
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
            logger.info(f"已用{weight}验证{len(losses)}个数据, Mean Loss: {ave_loss:.4f}, Mean PSNR: {ave_psnr:.2f}dB, Mean SSIM: {ave_ssim:.4f}, Mean MSE:{ave_mse:.4f}")
            logger.info(f'val数据集地址:{rcsdir}, 总耗时:{time.strftime("%H:%M:%S", time.gmtime(time.time()-tic))}')
            logger.info(f"损坏的文件：{corrupted_files}")
        logger.info(f'val数据集地址:{rcsdir}, 总耗时:{time.strftime("%H:%M:%S", time.gmtime(time.time()-tic))}')
        logger.info(f'↑----val loss:{ave_loss:.4f},psnr:{ave_psnr:.2f},ssim:{ave_ssim:.4f},mse:{ave_mse:.4f},inftime:{ave_inftime:.4f}s----↑')

        statisdir = os.path.join(save_dir,f'statistic_epoch{epoch}_PSNR{ave_psnr:.2f}dB_SSIM{ave_ssim:.4f}_MSE:{ave_mse:.4f}_Loss{ave_loss:.4f}.png')
        plotstatistic2(psnrs,ssims,mses,statisdir)
        savefigdata(psnrs,img_path=os.path.join(save_dir,f'sta/valall_epoch{epoch}psnrs{ave_psnr:.2f}.png'))
        savefigdata(ssims,img_path=os.path.join(save_dir,f'sta/valall_epoch{epoch}ssims{ave_ssim:.4f}.png'))
        savefigdata(mses,img_path=os.path.join(save_dir,f'sta/valall_epoch{epoch}mses{ave_mse:.4f}.png'))

        plot_scatter_separate(freq_list, mse_samples, psnr_samples, ssim_samples, save_dir)
        savefigdata(psnr_samples,img_path=os.path.join(save_dir,f'scatter_psnr_vs_freq.png'))
        savefigdata(ssim_samples,img_path=os.path.join(save_dir,f'scatter_ssim_vs_freq.png'))
        savefigdata(mse_samples,img_path=os.path.join(save_dir,f'scatter_mse_vs_freq.png'))
        savefigdata(freq_list,img_path=os.path.join(save_dir,f'scatter_vs_freq.png'))
    return ave_mse, ave_psnr, ave_ssim, psnrs, ssims, mses  #ave_psnr, 


if __name__ == '__main__':
    starttime = time.time()
    trainval = False
    cuda = 'cuda:1'
    draw = True
    draw = False
    draw3d = False
    lgrcs = False
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    valbatch = 20
    epoch = -1
    attnlayer = 1
    val_mse_per_plane = {}
    val_psnr_per_plane = {}
    val_ssim_per_plane = {}
    from datetime import datetime
    date = datetime.today().strftime("%m%d")
    save_dir = str(increment_path(Path(ROOT / "outputGNN" / "inference" / (f'{date}_valall_perfreq')), exist_ok=False))
    logdir = os.path.join(save_dir, 'alog.txt')
    logger = get_logger(logdir)
    valmsesavedir = os.path.join(save_dir, 'valmsecolume.png')
    valpsnrsavedir = os.path.join(save_dir, 'valpsnrcolume.png')
    valssimsavedir = os.path.join(save_dir, 'valssimcolume.png')

    datafolder = r'/mnt/truenas_jiangxiaotian/allplanes/mie'

    Fold1 = ['b871','bb7d','b827','b905','bbc6']
    Fold2 = ['b80b','ba0f','b7c1','b9e6','bb7c']
    Fold3 = ['b943','b97b','b812','bc2c','b974']
    Fold4 = ['bb26','b7fd','baa9','b979','b8ed']
    planes = ['b871', 'bb7d', 'b827', 'b905', 'bbc6', 'b80b', 'ba0f', 'b7c1', 'b9e6', 'bb7c', 'b943', 'b97b', 'b812', 'bc2c', 'b974', 'bb26', 'b7fd', 'baa9', 'b979', 'b8ed']
    # planes = ['b943','b7fd']

    trainplanes = ['b943']
    valplanes = None

    if valplanes is None:
        valplanes = list(set(planes) - set(trainplanes)) 
    if trainplanes is None:
        trainplanes = list(set(planes) - set(valplanes))  
    
    weight = r'last.pt'

    val_files = [plane + '_mie_val' for plane in planes]

    val_dataloaders = {}

    for valfile1 in val_files:
        valdataset = MultiEMRCSDataset([valfile1], datafolder)
        plane1 = valfile1[:4]
        val_dataloaders[plane1] = DataLoader.DataLoader(valdataset, batch_size=valbatch, shuffle=False, num_workers=16, pin_memory=True)

    for plane, valdataloader in val_dataloaders.items():
        logger.info(f"开始对飞机{plane}进行验证")
        valplanedir = os.path.join(save_dir, plane)
        if not os.path.exists(valplanedir):
            os.makedirs(valplanedir)
        logger.info(f'正在用{weight}验证推理{plane}及画图')
        valmse, valpsnr, valssim, valpsnrs, valssims, valmses = valmain(draw, device, weight, valfile1, valplanedir, logger, epoch, trainval, draw3d, attnlayer=attnlayer, valdataloader=valdataloader, batchsize=valbatch)
        val_mse_per_plane[plane] = valmse.item()
        val_psnr_per_plane[plane] = valpsnr.item()
        val_ssim_per_plane[plane] = valssim.item()

    train_mse_values = [val_mse_per_plane[plane] for plane in trainplanes]
    val_mse_values = [val_mse_per_plane[plane] for plane in valplanes]
    train_psnr_values = [val_psnr_per_plane[plane] for plane in trainplanes]
    val_psnr_values = [val_psnr_per_plane[plane] for plane in valplanes]
    train_ssim_values = [val_ssim_per_plane[plane] for plane in trainplanes]
    val_ssim_values = [val_ssim_per_plane[plane] for plane in valplanes]

    allavemse_train = np.mean(train_mse_values)
    allavemse_val = np.mean(val_mse_values)
    allavepsnr_train = np.mean(train_psnr_values)
    allavepsnr_val = np.mean(val_psnr_values)
    allavessim_train = np.mean(train_ssim_values)
    allavessim_val = np.mean(val_ssim_values)

    plt.clf()
    sorted_mse = sorted(val_mse_per_plane.items(), key=lambda x: x[1]) 
    planes_mse, mse_values = zip(*sorted_mse)

    bars = plt.bar(planes_mse, mse_values, label='MSE per Plane')
    plt.axhline(allavemse_train, color='darkseagreen', linestyle='--', label=f'Ave Held-Out Val MSE ({allavemse_train:.4f})')
    savefigdata(allavemse_train,img_path=os.path.join(save_dir, 'heldout_valmse.png'))
    plt.axhline(allavemse_val, color='lightskyblue', linestyle='--', label=f'Ave OOD Val MSE ({allavemse_val:.4f})')
    savefigdata(allavemse_val,img_path=os.path.join(save_dir, 'OOD_valmse.png'))
    plt.xlabel('Plane')
    plt.ylabel('MSE')
    plt.title('Validation MSE per Plane')
    legend_elements = [
        Line2D([0], [0], color='darkseagreen', lw=2, linestyle='--', label=f'Ave Held-Out Val MSE ({allavemse_train:.4f})'),
        Line2D([0], [0], color='lightskyblue', lw=2, linestyle='--', label=f'Ave OOD Val MSE ({allavemse_val:.4f})'),
        Rectangle((0, 0), 0.9, 0.5, color='darkseagreen', label='Train Plane'), 
        Rectangle((0, 0), 0.9, 0.5, color='lightskyblue', label='Val Plane') 
    ]
    plt.legend(handles=legend_elements, loc='best')
    # plt.legend()
    plt.xticks(rotation=80)

    planes_msedata = []
    for i, bar in enumerate(bars):
        if planes_mse[i] in trainplanes:
            bar.set_color('darkseagreen')  
            planes_msedata.append(('Train', planes_mse[i], mse_values[i])) 
        else:
            bar.set_color('lightskyblue') 
            planes_msedata.append(('Val', planes_mse[i], mse_values[i]))  

        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.4f}', ha='center', va='bottom', rotation=80)

    plt.ylim(0, max(mse_values) * 1.2)  
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(valmsesavedir)
    savefigdata(planes_msedata, img_path=valmsesavedir)
    plt.close()


    plt.clf()
    sorted_psnr = sorted(val_psnr_per_plane.items(), key=lambda x: x[1], reverse=True)  
    planes_psnr, psnr_values = zip(*sorted_psnr)

    bars = plt.bar(planes_psnr, psnr_values, label='PSNR per Plane')
    plt.axhline(allavepsnr_train, color='darkseagreen', linestyle='--', label=f'Ave Held-Out Val PSNR ({allavepsnr_train:.2f})')
    savefigdata(allavepsnr_train,img_path=os.path.join(save_dir, 'heldout_valpsnr.png'))
    plt.axhline(allavepsnr_val, color='lightskyblue', linestyle='--', label=f'Ave OOD Val PSNR ({allavepsnr_val:.2f})')
    savefigdata(allavepsnr_val,img_path=os.path.join(save_dir, 'OOD_valpsnr.png'))
    plt.xlabel('Plane')
    plt.ylabel('PSNR')
    plt.title('Validation PSNR per Plane')
    # plt.legend()
    legend_elements = [
        Line2D([0], [0], color='darkseagreen', lw=2, linestyle='--',label=f'Ave Held-Out Val PSNR ({allavepsnr_train:.2f})'),
        Line2D([0], [0], color='lightskyblue', lw=2, linestyle='--',label=f'Ave OOD Val PSNR ({allavepsnr_val:.2f})'),
        Rectangle((0, 0), 0.9, 0.5, color='darkseagreen', label='Train Plane'), 
        Rectangle((0, 0), 0.9, 0.5, color='lightskyblue', label='Val Plane') 
    ]
    plt.legend(handles=legend_elements, loc='best')
    plt.xticks(rotation=80)

    planes_psnrdata = []
    for i, bar in enumerate(bars):
        if planes_psnr[i] in trainplanes:
            bar.set_color('darkseagreen') 
            planes_psnrdata.append(('Train', planes_psnr[i], psnr_values[i])) 
        else:
            bar.set_color('lightskyblue')  
            planes_psnrdata.append(('Val', planes_psnr[i], psnr_values[i])) 

        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom', rotation=80)

    plt.ylim(0, max(psnr_values) * 1.2)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(valpsnrsavedir)
    savefigdata(planes_psnrdata, img_path=valpsnrsavedir)#('Train/Val', 'Plane name', MSE value)
    plt.close()



    plt.clf()
    sorted_ssim = sorted(val_ssim_per_plane.items(), key=lambda x: x[1], reverse=True)
    planes_ssim, ssim_values = zip(*sorted_ssim)

    bars = plt.bar(planes_ssim, ssim_values, label='SSIM per Plane')
    plt.axhline(allavessim_train, color='darkseagreen', linestyle='--', label=f'Ave Held-Out Val SSIM ({allavessim_train:.4f})')
    plt.axhline(allavessim_val, color='lightskyblue', linestyle='--', label=f'Ave OOD Val SSIM ({allavessim_val:.4f})')
    savefigdata(allavessim_train,img_path=os.path.join(save_dir, 'heldout_valssim.png'))
    savefigdata(allavessim_val,img_path=os.path.join(save_dir, 'OOD_valssim.png'))
    plt.xlabel('Plane')
    plt.ylabel('SSIM')
    plt.title('Validation SSIM per Plane')
    # plt.legend()
    legend_elements = [
        Line2D([0], [0], color='darkseagreen', lw=2, linestyle='--',label=f'Ave Held-Out Val SSIM ({allavessim_train:.4f})'),
        Line2D([0], [0], color='lightskyblue', lw=2, linestyle='--',label=f'Ave OOD Val SSIM ({allavessim_val:.4f})'),
        Rectangle((0, 0), 0.9, 0.5, color='darkseagreen', label='Train Plane'),
        Rectangle((0, 0), 0.9, 0.5, color='lightskyblue', label='Val Plane') 
    ]
    plt.legend(handles=legend_elements, loc='best')
    plt.xticks(rotation=80)

    planes_ssimdata = []
    for i, bar in enumerate(bars):
        if planes_ssim[i] in trainplanes:
            bar.set_color('darkseagreen')  
            planes_ssimdata.append(('Train', planes_ssim[i], ssim_values[i])) 
        else:
            bar.set_color('lightskyblue') 
            planes_ssimdata.append(('Val', planes_ssim[i], ssim_values[i])) 

        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.4f}', ha='center', va='bottom', rotation=80)

    plt.ylim(0, max(ssim_values) * 1.2)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(valssimsavedir)
    savefigdata(planes_ssimdata, img_path=valssimsavedir)

    plt.close()

    logger.info(f'总耗时：{(time.time()-starttime)/3600:.2f}小时或{(time.time()-starttime)/60:.2f}分钟')