from pathlib import Path
import os
import torch.utils.data.dataset as Dataset
import torch
import torch.nn.functional as F
import logging
import numpy as np
import trimesh
from net.data import derive_face_edges_from_faces

def savefigdata(*datas, img_path):
    """
    自动保存多个绘图数据到与图片同路径的 'data' 文件夹下，覆盖每次保存。
    示例：
    输入图片路径："/path/to/loss_curve.png"
    生成数据文件："/path/to/data/loss_curve_1.npy"、"/path/to/data/loss_curve_2.npy" 等
    """
    # 获取文件名（不含扩展名）
    base_data_name = os.path.splitext(os.path.basename(img_path))[0]  # 获取文件名部分，如 "loss_curve"
    
    # 获取数据保存目录并创建 'data' 文件夹
    data_dir = os.path.join(os.path.dirname(img_path), 'data')  # 使用图片所在目录加上 'data' 文件夹
    os.makedirs(data_dir, exist_ok=True)  # 创建 data 文件夹（如果不存在）

    # 逐个保存每个数据文件
    for idx, data in enumerate(datas):
        # 生成数据保存路径
        if idx == 0:
            data_path = os.path.join(data_dir, f"{base_data_name}.npy")  # 第一个数据直接保存为 loss_curve.npy
        else:
            data_path = os.path.join(data_dir, f"{base_data_name}_{idx + 1}.npy")  # 后续数据保存为 loss_curve_2.npy 等
        
        # 统一数据格式转换
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()

        # 保存为.npy格式，覆盖已存在的文件
        np.save(data_path, data)
        print(f"Saved data to: {data_path}")


class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *inputs):
        return self.model(
            vertices=inputs[0],
            faces=inputs[1],
            face_edges = inputs[2],
            # geoinfo=inputs[3],
            in_em=inputs[3],
            GT=inputs[4],
            device=inputs[5]
        )
    
def toc(tic):
    import time
    print(f'耗时{time.time() - tic:.4f}s')
    tic = time.time()
    return tic

def checksize(x):
    1
    # print(x.shape, torch.prod(torch.tensor(x.shape)).item())
    # return 1

# def ssim(img1, img2, window_size=11, size_average=True):
#     img1 = img1.unsqueeze(1) # [batch_size, 1, height, width]
#     img2 = img2.unsqueeze(1) # [batch_size, 1, height, width]
#     gtmax = torch.max(img1)
#     demax = torch.max(img2) #1.7787
#     maxx = torch.max(gtmax, demax)
#     C1 = (0.01 * maxx) ** 2
#     C2 = (0.03 * maxx) ** 2

#     def gaussian(window_size, sigma):
#         gauss = torch.exp(-(torch.arange(window_size) - window_size // 2) ** 2 / (2 * sigma ** 2))
#         return gauss / gauss.sum()

#     def create_window(window_size, channel):
#         _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
#         _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
#         window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
#         return window

#     channel = img1.size(0)
#     window = create_window(window_size, channel).to(img1.device)

#     mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
#     mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

#     mu1_sq = mu1.pow(2)
#     mu2_sq = mu2.pow(2)
#     mu1_mu2 = mu1 * mu2

#     sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
#     sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
#     sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

#     if size_average:
#         return ssim_map.mean()
#     else:
#         return ssim_map.mean(1).mean(1).mean(1)

def ssim(img1, img2, window_size=11, size_average=False):
    img1 = img1.unsqueeze(1)  # [batch_size, 1, height, width]
    img2 = img2.unsqueeze(1)  # [batch_size, 1, height, width]
    channel = img1.size(1)

    def gaussian(window_size, sigma):
        gauss = torch.exp(-(torch.arange(window_size).float() - window_size // 2) ** 2 / (2 * sigma ** 2))
        return gauss / gauss.sum()

    def create_window(window_size, channel):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    window = create_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    # 计算每张图片的最大值，并根据最大值计算C1和C2
    gtmax = img1.reshape(img1.size(0), -1).max(dim=1)[0]
    demax = img2.reshape(img2.size(0), -1).max(dim=1)[0]
    maxx = torch.max(gtmax, demax)
    C1 = (0.01 * maxx) ** 2
    C2 = (0.03 * maxx) ** 2

    # 将C1和C2调整为可广播的形状
    C1 = C1.view(-1, 1, 1, 1)
    C2 = C2.view(-1, 1, 1, 1)

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean([1, 2, 3])
    
def batch_ssim(img1, img2):
    ssim_val = ssim(img1, img2)
    return ssim_val
# def batch_ssim(img1, img2):
#     batch_size = img1.size(0)
#     ssim_list = []
#     for i in range(batch_size):
#         ssim_val = ssim(img1[i].unsqueeze(0), img2[i].unsqueeze(0))
#         ssim_list.append(ssim_val)
#     return torch.tensor(ssim_list)

# def psnr(img1, img2):
#     mse = F.mse_loss(img1, img2, reduction='mean')
#     gtmax = torch.max(img1)
#     demax = torch.max(img2)
#     maxx = torch.max(gtmax, demax)
#     psnr = 10 * torch.log10(maxx * maxx / mse)
#     return psnr

# def psnr(rcs, gt): #带gtmax的，不归一化，但是会出现rcsmax比gtmax大的情况
#     mse = F.mse_loss(rcs, gt, reduction='none')
#     mse = mse.view(mse.size(0), -1).mean(dim=1)  # Compute mean MSE for each image in the batch
#     maxx = gt.view(gt.size(0), -1).max(dim=1)[0]  # Max value of img2 for each image in batch
#     # maxx = torch.max(img1.view(img1.size(0), -1), dim=1)[0]
#     psnr = 10 * torch.log10(maxx ** 2 / mse)
#     return psnr

# def psnr(rcs, gt): #_with_dynamic_normalization 带gtmax的，用gtmax归一化，但是会出现rcsmax比gtmax大的情况，和上面其实是等价的
#     max1 = rcs.view(rcs.size(0), -1).max(dim=1)[0]  # Max value of img1 for each image in batch
#     max2 = gt.view(gt.size(0), -1).max(dim=1)[0]  # Max value of img2 for each image in batch
#     # max_val = torch.max(max1, max2)  # Element-wise max between max1 and max2
#     max_val = max2  # Element-wise max between max1 and max2
#     img1_normalized = rcs / max_val.view(-1, 1, 1, 1)
#     img2_normalized = gt / max_val.view(-1, 1, 1, 1)
#     mse = F.mse_loss(img1_normalized, img2_normalized, reduction='none')
#     mse = mse.view(mse.size(0), -1).mean(dim=1)  # Compute mean MSE for each image in the batch
#     psnr = 10 * torch.log10(1.0 / mse)  # Since normalized images have max_val = 1.0
#     return psnr

def psnr(img1, img2): #_with_dynamic_normalization 不带gtmax的，用两个的最大值归一化 已经考虑了逐张做
    max1 = img1.reshape(img1.size(0), -1).max(dim=1)[0]  # Max value of img1 for each image in batch
    max2 = img2.reshape(img2.size(0), -1).max(dim=1)[0]  # Max value of img2 for each image in batch
    max_val = torch.max(max1, max2)  # Element-wise max between max1 and max2
    img1_normalized = img1 / max_val.view(-1, 1, 1, 1)
    img2_normalized = img2 / max_val.view(-1, 1, 1, 1)

    mse = F.mse_loss(img1_normalized, img2_normalized, reduction='none')
    mse = mse.view(mse.size(0), -1).mean(dim=1)  # Compute mean MSE for each image in the batch
    psnr = 10 * torch.log10(1.0 / mse)  # Since normalized images have max_val = 1.0
    return psnr

def batch_psnr(img1, img2):
    psnrr=psnr(img1,img2)
    return psnrr

def batch_mse(img1, img2):
    return (img1.sub(img2).pow_(2).view(img1.size(0), -1).mean(1))
# def batch_psnr(img1, img2):
#     import time
#     tic = time.time()
#     batch_size = img1.size(0)
#     psnrlist=[]
#     for i in range(batch_size):
#         psnrr=psnr(img1[i],img2[i])
#         psnrlist.append(psnrr)
#     return torch.tensor(psnrlist)

def transform_to_log_coordinates(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    log_x = (torch.log10(torch.where(x <= 0, torch.finfo(torch.float).eps, x)) - torch.log10(torch.tensor(0.001))) / (torch.log10(torch.tensor(1.1)) - torch.log10(torch.tensor(0.001))) # 将负值和零转换为正值,取对数,归一化处理 现在是0-1而不是0-8了~！
    return log_x

def get_model_memory(model,logger):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    memory = params * 4 / (1024**3)
    logger.info(f'model memory{memory:.4f}GB')  # 以GB为单位
    # print(f'模型占用{memory:.4f}GB')  # 以GB为单位
    return memory

def get_model_memory_nolog(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    memory = params * 4 / (1024**3)
    # logger.info(f'模型占用{memory:.4f}GB')  # 以GB为单位
    print(f'模型占用{memory:.4f}GB')  # 以GB为单位
    return memory

def get_tensor_memory(x,logger):
    size = x.element_size() * x.nelement()
    MBmemory = size / (1024**2)
    GBmemory = size / (1024**3)
    logger.info(f'Tensor变量占用{GBmemory:.2f}GB或{MBmemory:.2f}MB')  # 以GB为单位
    # print(f'Tensor变量占用{memory:.4f}GB')  # 以GB为单位
    return GBmemory, MBmemory

def get_x_memory(x,logger):
    import sys
    size = sys.getsizeof(x)
    MBmemory = size / (1024**2)
    GBmemory = size / (1024**3)
    logger.info(f'Tensor变量占用{GBmemory:.2f}GB或{MBmemory:.2f}MB')  # 以GB为单位


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger

def increment_path(path, exist_ok=False, sep="", mkdir=True):
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path

class EMRCSDataset(Dataset.Dataset):
    def __init__(self, filelist, rcsdir):    #初始化，定义数据内容和标签
        self.filelist = filelist
        self.rcsdir = rcsdir
    def __len__(self):    #返回数据集大小
        return len(self.filelist)
    def __getitem__(self, index):    #得到数据内容和标签
        import re
        file = self.filelist[index]
        plane, theta, phi, freq= re.search(r"([a-zA-Z0-9]{4})_theta(\d+)phi(\d+)f(\d.+).pt", file).groups()
        theta = int(theta)
        phi = int(phi)
        freq = float(freq)
        in_em = [plane,theta,phi,freq]
        rcs = torch.load(os.path.join(self.rcsdir,file), weights_only=False)
        return in_em, rcs

class MultiEMRCSDataset(Dataset.Dataset):
    def __init__(self, folder_list, base_dir):  # 初始化，传入文件夹列表和基础目录
        self.folder_list = folder_list
        self.base_dir = base_dir
        self.filelist = []
        
        # 遍历每个文件夹，获取所有文件
        for folder in folder_list:
            folder_path = os.path.join(base_dir, folder)
            files = os.listdir(folder_path)
            self.filelist.extend([os.path.join(folder, file) for file in files])
    
    def __len__(self):  # 返回数据集大小
        return len(self.filelist)
    
    def __getitem__(self, index):  # 得到数据内容和标签
        import re
        file = self.filelist[index]
        full_path = os.path.join(self.base_dir, file)
        plane, theta, phi, freq = re.search(r"([a-zA-Z0-9]{4})_theta(\d+)phi(\d+)f(\d.+).pt", file).groups()
        theta = int(theta)
        phi = int(phi)
        freq = float(freq)
        in_em = [plane, theta, phi, freq]
        rcs = torch.load(full_path, weights_only=False)
        return in_em, rcs
    
def find_matching_files(prefixlist, directory):
    objlist = []
    ptlist = []
    for prefix in prefixlist:
        for filename in os.listdir(directory):
            if filename[:4] == prefix and filename[-4:]=='.obj':
                plane = filename[:-4]
                loadobj = plane + '.obj'
                loadpt = plane + '.pt'
                objlist.append(os.path.join(directory, loadobj))
                ptlist.append(os.path.join(directory, loadpt))
                break
    return objlist , ptlist


def load_and_process(file_path):
    """
    读取并处理单个文件，返回faces, verts, faceedge张量。pt是edge文件，obj是原始目标文件
    """
    mesh = trimesh.load_mesh(file_path)
    area = mesh.area
    volume = mesh.volume
    scale = mesh.scale
    faces = torch.tensor(mesh.faces, dtype=torch.int).unsqueeze(0)
    verts = torch.tensor(mesh.vertices, dtype=torch.float32).unsqueeze(0)
    pt_path = file_path.replace('.obj', '.pt')
    # faceedge = torch.load(pt_path)
    if os.path.exists(pt_path):
        faceedge = torch.load(pt_path)
    else:
        # faceedge = pt_path
        print('正在生成edge')
        faceedge = derive_face_edges_from_faces(faces, pad_id = -1)
        torch.save(faceedge, pt_path)
        print(f'已生成edge并保存到{pt_path}')
    return faces, verts, faceedge, [area, volume, scale]

def process_files(file_paths, device='cpu'):
    """
    处理一组文件并返回批量张量。
    
    Args:
    - file_paths (list of str): 文件路径列表。
    - device (str or torch.device): 使用的设备，默认为'cpu'。
    
    Returns:
    - planesur_faces (torch.Tensor): faces批量张量。
    - planesur_verts (torch.Tensor): verts批量张量。
    - planesur_faceedges (torch.Tensor): faceedges批量张量。
    """
    # 设置设备
    device = torch.device(device if isinstance(device, str) else device)

    # 读取所有文件并合并到batch
    faces_list, verts_list, faceedges_list, geoinfo_list = [], [], [], []
    for file_path in file_paths:
        faces, verts, faceedge, geoinfo = load_and_process(file_path)
        faces_list.append(faces)
        verts_list.append(verts)
        faceedges_list.append(faceedge)
        geoinfo_list.append(geoinfo)

    # 找到需要填充到的最大大小 完成了padding
    max_faces_size = max(f.shape[1] for f in faces_list)
    max_verts_size = max(v.shape[1] for v in verts_list)
    max_faceedges_size = max(fe.shape[1] for fe in faceedges_list)

    # 填充所有张量到相同大小
    padded_faces_list = [F.pad(f, (0, 0, 0, max_faces_size - f.shape[1])) for f in faces_list]
    padded_verts_list = [F.pad(v, (0, 0, 0, max_verts_size - v.shape[1])) for v in verts_list]
    padded_faceedges_list = [F.pad(fe, (0, 0, 0, max_faceedges_size - fe.shape[1])) for fe in faceedges_list]

    # 合并到batch
    planesur_faces = torch.cat(padded_faces_list, dim=0).to(device)
    planesur_verts = torch.cat(padded_verts_list, dim=0).to(device)
    planesur_faceedges = torch.cat(padded_faceedges_list, dim=0).to(device)

    return planesur_faces, planesur_verts, planesur_faceedges, geoinfo_list

