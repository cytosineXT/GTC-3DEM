# from pathlib import Path
from functools import partial
from math import pi
import time
import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torchtyping import TensorType

from beartype import beartype
from beartype.typing import  Tuple, Optional

from einops import rearrange, reduce, pack
from einops.layers.torch import Rearrange

from net.utils_newload import transform_to_log_coordinates, psnr, batch_mse, derive_face_edges_from_faces
from net.utils_newload import ssim as myssim
from pytorch_msssim import ms_ssim, ssim

from torch_geometric.nn.conv import SAGEConv
import numpy as np

class LearnableMedianFilter(nn.Module):
    def __init__(self, kernel_size=5):
        super(LearnableMedianFilter, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, img):
        """应用中值滤波"""
        pad_size = self.kernel_size // 2  # 计算 padding 大小
        img_padded = F.pad(img, (pad_size, pad_size, pad_size, pad_size), mode='reflect')  # 对图像进行 padding
        batch_size, channels, height, width = img_padded.shape  # 获取图像的尺寸
        unfolded = F.unfold(img_padded, kernel_size=self.kernel_size)  # 展开图像矩阵
        unfolded = unfolded.view(batch_size, channels, self.kernel_size * self.kernel_size, -1)  # 计算中值
        median = unfolded.median(dim=2)[0]
        median = median.view(batch_size, channels, height - 2 * pad_size, width - 2 * pad_size)  # 恢复图像尺寸
        return median
class LearnableGaussianFilter(nn.Module):
    def __init__(self, kernel_size=5, init_sigma=4.0, device='cuda:0'):
        super(LearnableGaussianFilter, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = nn.Parameter(torch.tensor(init_sigma, dtype=torch.float32))  # 可学习的 sigma
        self.device = device

    def gaussian_kernel(self):
        """生成一个高斯核"""
        size = self.kernel_size
        sigma = self.sigma
        kernel = torch.tensor([[(1/(2.0*np.pi*sigma**2)) * torch.exp(-((x - size//2)**2 + (y - size//2)**2)/(2*sigma**2))
                               for x in range(size)] for y in range(size)]).float().to(self.device)
        kernel /= kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)

    def forward(self, img):
        """应用高斯滤波"""
        kernel = self.gaussian_kernel()
        channels = img.shape[1]
        kernel = kernel.repeat(channels, 1, 1, 1)
        padding = self.kernel_size // 2
        filtered_img = F.conv2d(img, kernel, padding=padding, groups=channels)
        return filtered_img
class SmoothingLayer(nn.Module):
    def __init__(self, kernel_size=5, init_sigma=4.0, device='cuda:0'):
        super(SmoothingLayer, self).__init__()
        self.median_filter = LearnableMedianFilter(kernel_size=kernel_size)
        self.gaussian_filter = LearnableGaussianFilter(kernel_size=kernel_size, init_sigma=init_sigma, device=device)

    def forward(self, img):
        """应用中值滤波和高斯滤波"""
        img = img.unsqueeze(1)  # 添加 channel 维度
        img = self.median_filter(img)  # 应用中值滤波
        img = self.gaussian_filter(img)  # 应用高斯滤波
        img = img.squeeze(1)  # 移除 channel 维度
        return img

def median_filter2d(img, kernel_size=5):
    pad_size = kernel_size // 2    # 计算 padding 大小
    img_padded = F.pad(img, (pad_size, pad_size, pad_size, pad_size), mode='reflect')    # 对图像进行 padding
    batch_size, channels, height, width = img_padded.shape    # 获取图像的尺寸
    unfolded = F.unfold(img_padded, kernel_size=kernel_size)    # 展开图像矩阵
    unfolded = unfolded.view(batch_size, channels, kernel_size * kernel_size, -1)    # 计算中值
    median = unfolded.median(dim=2)[0]
    median = median.view(batch_size, channels, height - 2 * pad_size, width - 2 * pad_size)    # 恢复图像尺寸
    return median

def gaussian_kernel(size, sigma, device):
    """生成一个高斯核"""
    kernel = torch.tensor([[(1/(2.0*np.pi*sigma**2)) * np.exp(-((x - size//2)**2 + (y - size//2)**2)/(2*sigma**2))
                            for x in range(size)] for y in range(size)]).float().to(device)
    kernel /= kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0)

def gaussian_filter2d(img, kernel_size=5, sigma=4.0, device='cuda:0'):
    """应用高斯滤波"""
    kernel = gaussian_kernel(kernel_size, sigma, device)
    channels = img.shape[1]
    kernel = kernel.repeat(channels, 1, 1, 1)
    padding = kernel_size // 2
    filtered_img = F.conv2d(img, kernel, padding=padding, groups=channels)
    return filtered_img

def loss_fn(decoded, GT, loss_type='L1', gama=0.01, delta=0.5):

    maxloss = torch.mean(torch.abs(torch.amax(decoded, dim=(1, 2)) - torch.amax(GT, dim=(1, 2))))
    minus = decoded - GT
    mse = ((minus) ** 2).mean() #111 就是一样的
    nmse = mse / torch.var(GT)
    rmse = torch.sqrt(mse)
    l1 = (decoded-GT).abs().mean()
    # percentage_error = (minus / (GT + 1e-2)).abs().mean()
    percentage_error = (minus / (GT + 1e-8)).abs().mean()

    if loss_type == 'mse':
        # loss = F.mse_loss(decoded, GT)
        loss = mse
    elif loss_type == 'L1':
        # loss = F.l1_loss(pred, target)
        loss = l1
    elif loss_type == 'rmse':
        loss = rmse
    elif loss_type == 'nmse':
        loss = nmse
    elif loss_type == 'per':
        loss = percentage_error
    elif loss_type == 'ssim':
        loss = 1 - torch.stack([ssim(decoded[i].unsqueeze(0).unsqueeze(0), GT[i].unsqueeze(0).unsqueeze(0), data_range=max(decoded[i].max().item(), GT[i].max().item()), size_average=False) for i in range(decoded.size(0))]).squeeze().mean()
    elif loss_type == 'msssim':
        loss = 1 - torch.stack([ms_ssim(decoded[i].unsqueeze(0).unsqueeze(0), GT[i].unsqueeze(0).unsqueeze(0), data_range=max(decoded[i].max().item(), GT[i].max().item()), size_average=False) for i in range(decoded.size(0))]).squeeze().mean()

    elif loss_type == 'mse_l1':
        loss = delta * mse + (1 - delta) * l1
    elif loss_type == 'mse_nmse':
        loss = delta * mse + (1 - delta) * nmse
    elif loss_type == 'l1_nmse':
        loss = delta * l1 + (1 - delta) * nmse
    elif loss_type == 'mse_ssim':
        ssim_val = torch.stack([ssim(decoded[i].unsqueeze(0).unsqueeze(0), GT[i].unsqueeze(0).unsqueeze(0), data_range=max(decoded[i].max().item(), GT[i].max().item()), size_average=False) for i in range(decoded.size(0))]).squeeze().mean()
        loss = delta * mse + (1 - delta) * (1 - ssim_val)
    elif loss_type == 'mse_msssim':
        msssim_val = torch.stack([ms_ssim(decoded[i].unsqueeze(0).unsqueeze(0), GT[i].unsqueeze(0).unsqueeze(0), data_range=max(decoded[i].max().item(), GT[i].max().item()), size_average=False) for i in range(decoded.size(0))]).squeeze().mean()
        loss = delta * mse + (1 - delta) * (1 - msssim_val)
    elif loss_type == 'l1_ssim':
        ssim_val = torch.stack([ssim(decoded[i].unsqueeze(0).unsqueeze(0), GT[i].unsqueeze(0).unsqueeze(0), data_range=max(decoded[i].max().item(), GT[i].max().item()), size_average=False) for i in range(decoded.size(0))]).squeeze().mean()
        loss = delta * l1 + (1 - delta) * (1 - ssim_val)
    elif loss_type == 'l1_msssim':
        msssim_val = torch.stack([ms_ssim(decoded[i].unsqueeze(0).unsqueeze(0), GT[i].unsqueeze(0).unsqueeze(0), data_range=max(decoded[i].max().item(), GT[i].max().item()), size_average=False) for i in range(decoded.size(0))]).squeeze().mean()
        loss = delta * l1 + (1 - delta) * (1 - msssim_val)
    else:
        print(f"Unsupported loss type: {loss_type}, will use l1 loss")
        loss = l1

    total_loss = loss + gama * maxloss
    return total_loss

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def first(it):
    return it[0]

def divisible_by(num, den):
    return (num % den) == 0

def is_odd(n):
    return not divisible_by(n, 2)

def is_empty(l):
    return len(l) == 0

def is_tensor_empty(t: Tensor):
    return t.numel() == 0

def set_module_requires_grad_(
    module: Module,
    requires_grad: bool
):
    for param in module.parameters():
        param.requires_grad = requires_grad

def l1norm(t):
    return F.normalize(t, dim = -1, p = 1)

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

def safe_cat(tensors, dim):
    tensors = [*filter(exists, tensors)]

    if len(tensors) == 0:
        return None
    elif len(tensors) == 1:
        return first(tensors)

    return torch.cat(tensors, dim = dim)

def pad_at_dim(t, padding, dim = -1, value = 0):
    ndim = t.ndim
    right_dims = (ndim - dim - 1) if dim >= 0 else (-dim - 1)
    zeros = (0, 0) * right_dims
    return F.pad(t, (*zeros, *padding), value = value)

def pad_to_length(t, length, dim = -1, value = 0, right = True):
    curr_length = t.shape[dim]
    remainder = length - curr_length

    if remainder <= 0:
        return t

    padding = (0, remainder) if right else (remainder, 0)
    return pad_at_dim(t, padding, dim = dim, value = value)

# continuous embed

def ContinuousEmbed(dim_cont):
    return nn.Sequential(
        Rearrange('... -> ... 1'),
        nn.Linear(1, dim_cont),
        nn.SiLU(),
        nn.Linear(dim_cont, dim_cont),
        nn.LayerNorm(dim_cont)
    )

# additional encoder features
# 1. angle (3), 2. area (1), 3. normals (3)  四、入射波入射角（1）or 入射矢量(3) 五、入射波频率

@torch.no_grad()
def coords_interanglejxt2(x, y, eps=1e-5): #不要用爱恩斯坦求和 会变得不幸
    edge_vector = x - y
    normv = l2norm(edge_vector) #torch.Size([2, 20804, 3, 3])
    normdot = -(normv * torch.cat((normv[..., -1:], normv[..., :-1]), dim=3)).sum(dim=2) #应为torch.Size([2, 20804])
    normdot = torch.clamp(normdot, -1 + eps, 1 - eps)
    radians = torch.acos(normdot) #tensor([1.1088, 0.8747, 1.1511], device='cuda:0')
    angle = torch.rad2deg(radians) #tensor([63.5302, 50.1188, 65.9518], device='cuda:0')
    return radians, angle

@torch.no_grad()
def vector_anglejxt2(x, y, eps=1e-5):
    normdot = -(l2norm(x) * l2norm(y)).sum(dim=-1)
    normdot = torch.clamp(normdot, -1 + eps, 1 - eps)
    radians = normdot.acos() #tensor(1.4104, device='cuda:0')
    angle = torch.rad2deg(radians) #tensor(80.8117, device='cuda:0')
    return radians, angle

@torch.no_grad()
def polar_to_cartesian2(theta, phi):
    theta_rad = torch.deg2rad(theta)
    phi_rad = torch.deg2rad(phi)
    x = torch.sin(phi_rad) * torch.cos(theta_rad)
    y = torch.sin(phi_rad) * torch.sin(theta_rad)
    z = torch.cos(phi_rad)
    return torch.stack([x, y, z], dim=1)

@torch.no_grad()
def jxtget_face_coords(vertices, face_indices):
    batch_size, num_faces, num_vertices_per_face = face_indices.shape
    reshaped_face_indices = face_indices.reshape(batch_size, -1).to(dtype=torch.int64) 
    face_coords = torch.gather(vertices, 1, reshaped_face_indices.unsqueeze(-1).expand(-1, -1, vertices.shape[-1])) # 使用索引张量获取具有坐标的面
    face_coords = face_coords.reshape(batch_size, num_faces, num_vertices_per_face, -1)# 还原形状
    return face_coords

@torch.no_grad()
def get_derived_face_featuresjxt(
    face_coords: TensorType['b', 'nf', 'nvf', 3, float],  # 3 or 4 vertices with 3 coordinates输入坐标格式的face list
    in_em, #\theta, \phi, ka
    device
):

    shifted_face_coords = torch.cat((face_coords[:, :, -1:], face_coords[:, :, :-1]), dim = 2).to(device) #这是对face_coords循环移位，face_coords[:, :, -1:]取最后一个切片，
    angles, _  = coords_interanglejxt2(face_coords, shifted_face_coords) #得到了每个三角形face的三个内角，弧度形式的，如果要角度形式的要用_的(angle2) 耗时1.64！
    edge1, edge2, *_ = (face_coords - shifted_face_coords).unbind(dim = 2) #这里是坐标相减得到边
    normals = l2norm(torch.cross(edge1, edge2, dim = -1)) #然后也用边叉乘得到法向量，很合理
    area = torch.cross(edge1, edge2, dim = -1).norm(dim = -1, keepdim = True) * 0.5 #两边矢量叉乘模/2得到面积
    incident_angle_vec = polar_to_cartesian2(in_em[1],in_em[2]) #得到入射方向的xyz矢量
    # incident_angle_vec = polar_to_cartesian2(in_em[:,1],in_em[:,2]) #得到入射方向的xyz矢量
    incident_angle_mtx = incident_angle_vec.unsqueeze(1).repeat(1, area.shape[1], 1).to(device) #得到入射方向的矢量矩阵torch.Size([batchsize, 33564, 3])
    incident_freq_mtx = in_em[3].unsqueeze(1).unsqueeze(2).repeat(1, area.shape[1], 1).to(device) #得到入射波频率的矩阵torch.Size([1, 33564, 1]) 
    # incident_freq_mtx = in_em[:,3].unsqueeze(1).unsqueeze(2).repeat(1, area.shape[1], 1) #得到入射波频率的矩阵torch.Size([1, 33564, 1]) 
    incident_mesh_anglehudu, _ = vector_anglejxt2(normals, incident_angle_mtx) #得到入射方向和每个mesh法向的夹角,是在0到180度的，0-90说明面在物体屁股后面，90-180说明是正面  耗时0.68！！！
    return dict(
        angles = angles,
        area = area,
        normals = normals,
        emnoangle = incident_mesh_anglehudu,
        emangle = incident_angle_mtx,
        emfreq = incident_freq_mtx
    ) , incident_angle_vec #这里就算回了freq=em[0][2]好像也没啥用吧，没离散化的 入射方向矢量倒是有用！

@beartype
def discretize(
    t: Tensor,
    *, #*表示后面的参数都必须以关键字参数的形式传递，也就是说，在调用这个函数时，必须明确指定参数的名称，例如discretize(t, continuous_range=(0.0, 1.0), num_discrete=128)。
    continuous_range: Tuple[float, float],
    num_discrete: int = 128
) -> Tensor: 
    lo, hi = continuous_range
    assert hi > lo
    t = (t - lo) / (hi - lo)
    t *= num_discrete
    t -= 0.5
    return t.round().long().clamp(min = 0, max = num_discrete - 1) #.round四舍五入 .long转为长整形 .clamp限制在min和max中间(一层鲁棒保险)


class MeshCodec(Module):
    @beartype
    def __init__(
        self,
        num_discrete_coors = 512,   #坐标离散量改成512
        coor_continuous_range: Tuple[float, float] = (-1., 1.), #连续坐标范围
        dim_coor_embed = 64,        #坐标embedding维度
        num_discrete_area = 128,    #面积离散量
        dim_area_embed = 16,        #面积embedding维度
        num_discrete_normals = 128, #法线离散量
        dim_normal_embed = 64,      #法线embedding维度
        num_discrete_angle = 128,   #角度离散量
        dim_angle_embed = 16,       #角度embedding维度
        num_discrete_emnoangle = 128,   #法线、入射夹角离散量 jxt
        dim_emnoangle_embed = 16,       #法线、入射夹角embedding维度 jxt
        num_discrete_emangle = 128,     #入射角离散量 jxt
        dim_emangle_embed = 64,         #入射角embedding维度 jxt
        num_discrete_emfreq = 512,      #频率(Ka)离散量 jxt 2024年5月11日15:51:05 128改为512
        dim_emfreq_embed = 16,          #频率(Ka)embedding维度 jxt
        encoder_dims_through_depth: Tuple[int, ...] = (
            64, 128, 256, 256, 576
        ),                          #encoder每一层的维度
        dim_codebook = 192,         #codebook维度
        sageconv_kwargs: dict = dict(   #SAGEconv参数
            normalize = True,
            project = True
        ),
        attn_encoder_depth = 0,     #encoder注意力深度层数
        local_attn_kwargs: dict = dict( #局部注意力参数
            dim_head = 32,
            heads = 8
        ),
        local_attn_window_size = 64,    #局部注意力窗口大小
        linear_attn_kwargs: dict = dict(    #线性注意力参数
            dim_head = 8,
            heads = 16
        ),
        use_linear_attn = True,     #是否使用线性注意力
        pad_id = -1,                #填充id，这是在处理变长面时使用的填充值
        attn_dropout = 0.,          #注意力层dropout率
        ff_dropout = 0.,            #前馈网络dropout率
        quads = False,               #是否使用四边形
        middim = 64,
        device = 'cpu',
        paddingsize = 22500,
    ): 
        super().__init__()

        self.conv1d1 = nn.Conv1d(576, middim, kernel_size=10, stride=10, dilation=1 ,padding=0)
        # self.conv1d1 = nn.Conv1d(784, middim, kernel_size=10, stride=10, dilation=1 ,padding=0)
        self.fc1d1 = nn.Linear(2250, 45*90)
        # self.conv1d1 = nn.Conv1d(784, 1, kernel_size=10, stride=10, dilation=1 ,padding=0)
        # self.fc1d1 = nn.Linear(2250, middim*45*90)

        # torch.Size([10, 64(middim), 2250])
        self.incident_angle_linear1 = nn.Linear(2, 2250)
        self.emfreq_embed1 = nn.Embedding(num_discrete_emfreq, 2250)
        self.incident_angle_linear2 = nn.Linear(2, 4050)
        self.emfreq_embed2 = nn.Embedding(num_discrete_emfreq, 4050)
        self.incident_angle_linear3 = nn.Linear(2, 90*180)
        self.emfreq_embed3 = nn.Embedding(num_discrete_emfreq, 90*180)
        self.incident_angle_linear4 = nn.Linear(2, 180*360)
        self.emfreq_embed4 = nn.Embedding(num_discrete_emfreq, 180*360)
        self.incident_angle_linear5 = nn.Linear(2, 360*720)
        self.emfreq_embed5 = nn.Embedding(num_discrete_emfreq, 360*720)

        self.smoothing_layer = SmoothingLayer(kernel_size=5, init_sigma=4.0, device=device)

        # Decoder3
        self.upconv1 = nn.ConvTranspose2d(middim, int(middim/2), kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(int(middim/2))  # 添加的批量归一化层1
        self.conv1_1 = nn.Conv2d(int(middim/2), int(middim/2), kernel_size=3, stride=1, padding=1)  # 添加的卷积层1
        self.conv1_2 = nn.Conv2d(int(middim/2), int(middim/2), kernel_size=3, stride=1, padding=1)  # 添加的卷积层2
        self.bn1_1 = nn.BatchNorm2d(int(middim/2))  # 添加的批量归一化层1
        self.bn1_2 = nn.BatchNorm2d(int(middim/2))  # 添加的批量归一化层2
        self.upconv2 = nn.ConvTranspose2d(int(middim/2), int(middim/4), kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(int(middim/4))  # 添加的批量归一化层1
        self.conv2_1 = nn.Conv2d(int(middim/4), int(middim/4), kernel_size=3, stride=1, padding=1)  # 添加的卷积层1
        self.conv2_2 = nn.Conv2d(int(middim/4), int(middim/4), kernel_size=3, stride=1, padding=1)  # 添加的卷积层2
        self.bn2_1 = nn.BatchNorm2d(int(middim/4))  # 添加的批量归一化层1
        self.bn2_2 = nn.BatchNorm2d(int(middim/4))  # 添加的批量归一化层2
        self.upconv3 = nn.ConvTranspose2d(int(middim/4), int(middim/8), kernel_size=2, stride=2)
        # self.upconv3 = nn.ConvTranspose2d(int(middim/4), int(middim/8), kernel_size=2, stride=2, output_padding=1)
        self.bn3 = nn.BatchNorm2d(int(middim/8))
        self.conv3_1 = nn.Conv2d(int(middim/8), int(middim/8), kernel_size=3, stride=1, padding=1)  # 添加的卷积层1
        self.conv3_2 = nn.Conv2d(int(middim/8), int(middim/8), kernel_size=3, stride=1, padding=1)  # 添加的卷积层1
        self.bn3_1 = nn.BatchNorm2d(int(middim/8))  # 添加的批量归一化层1
        self.bn3_2 = nn.BatchNorm2d(int(middim/8))  # 添加的批量归一化层2
        self.conv1x1 = nn.Conv2d(int(middim/8), 1, kernel_size=1, stride=1, padding=0)   #1×1卷积，把多的维度融合了


        self.num_vertices_per_face = 3 if not quads else 4

        self.num_discrete_coors = num_discrete_coors
        self.coor_continuous_range = coor_continuous_range
        self.discretize_face_coords = partial(discretize, num_discrete = num_discrete_coors, continuous_range = coor_continuous_range)
        self.coor_embed = nn.Embedding(num_discrete_coors, dim_coor_embed) 
        self.discretize_angle = partial(discretize, num_discrete = num_discrete_angle, continuous_range = (0., pi))
        self.angle_embed = nn.Embedding(num_discrete_angle, dim_angle_embed)
        lo, hi = coor_continuous_range
        self.discretize_area = partial(discretize, num_discrete = num_discrete_area, continuous_range = (0., (hi - lo) ** 2))
        self.area_embed = nn.Embedding(num_discrete_area, dim_area_embed)
        self.discretize_normals = partial(discretize, num_discrete = num_discrete_normals, continuous_range = coor_continuous_range)
        self.normal_embed = nn.Embedding(num_discrete_normals, dim_normal_embed)
        self.discretize_emnoangle = partial(discretize, num_discrete = num_discrete_emnoangle, continuous_range = (0., pi)) #0还是-pi? 是0到pi!! jxt
        self.emnoangle_embed = nn.Embedding(num_discrete_emnoangle, dim_emnoangle_embed) #jxt
        self.discretize_emangle = partial(discretize, num_discrete = num_discrete_emangle, continuous_range = coor_continuous_range) #jxt
        self.emangle_embed = nn.Embedding(num_discrete_emangle, dim_emangle_embed) #jxt
        self.discretize_emfreq = partial(discretize, num_discrete = num_discrete_emfreq, continuous_range = (0.,1.0)) 
        self.emfreq_embed = nn.Embedding(num_discrete_emfreq, dim_emfreq_embed) #jxt

        attn_kwargs = dict(
            causal = False,
            prenorm = True,
            dropout = attn_dropout,
            window_size = local_attn_window_size,
        )

        init_dim = dim_coor_embed * (3 * self.num_vertices_per_face) + dim_angle_embed * self.num_vertices_per_face + dim_normal_embed * 3 + dim_area_embed + dim_emangle_embed * 3 + dim_emnoangle_embed + dim_emfreq_embed

        self.project_in = nn.Linear(init_dim, dim_codebook)

        sageconv_kwargs = {**sageconv_kwargs }

        init_encoder_dim, *encoder_dims_through_depth = encoder_dims_through_depth #64, 128, 256, 256, 576
        curr_dim = init_encoder_dim

        self.init_sage_conv = SAGEConv(dim_codebook, init_encoder_dim, **sageconv_kwargs)

        self.init_encoder_act_and_norm = nn.Sequential(
            nn.SiLU(),
            nn.LayerNorm(init_encoder_dim)
        ) #所以是要手动激活？

        self.discretize_emfreq2 = partial(discretize, num_discrete = 512, continuous_range = (0.,1.0))
        self.condfreqlayers = ModuleList([ #长度不定没关系，我可以变成固定的维度让他在长度上广播！
            nn.Embedding(512, 64),
            nn.Embedding(512, 128), 
            nn.Embedding(512, 256), 
            nn.Embedding(512, 256),]
            # nn.Embedding(512, 1125),
            # nn.Embedding(512, 576),]
        )
        self.condanglelayers = ModuleList([
            nn.Linear(2, 64),
            nn.Linear(2, 128),
            nn.Linear(2, 256),
            nn.Linear(2, 256),]
            # nn.Linear(2, 1125),
            # nn.Linear(2, 576),]
        )

        self.encoders = ModuleList([])

        # for dim_layer in encoder_dims_through_depth:
        #     sage_conv = SAGEConv(
        #         curr_dim,
        #         dim_layer,
        #         **sageconv_kwargs
        #     )
        #     #需不需要手动加激活函数
        #     self.encoders.append(sage_conv) #这里把encoder创好了并贴上了sage
        #     curr_dim = dim_layer

        self.encoders = ModuleList([])
        self.encoder_act_and_norm = ModuleList([])  # 新增的激活和归一化层列表

        for dim_layer in encoder_dims_through_depth:
            sage_conv = SAGEConv(
                curr_dim,
                dim_layer,
                **sageconv_kwargs
            )
            self.encoders.append(sage_conv)  # 添加SAGEConv层
            # 添加激活函数和LayerNorm
            self.encoder_act_and_norm.append(nn.Sequential(
                nn.SiLU(),
                nn.LayerNorm(dim_layer)
            ))
            curr_dim = dim_layer
            
        # self.encoder_attn_blocks = ModuleList([])

        # for _ in range(attn_encoder_depth):
        #     self.encoder_attn_blocks.append(nn.ModuleList([
        #         TaylorSeriesLinearAttn(curr_dim, prenorm = True, **linear_attn_kwargs) if use_linear_attn else None,
        #         LocalMHA(dim = curr_dim, **attn_kwargs, **local_attn_kwargs),
        #         nn.Sequential(RMSNorm(curr_dim), FeedForward(curr_dim, glu = True, dropout = ff_dropout))
        #     ]))

        self.project_dim_codebook = nn.Linear(curr_dim, dim_codebook * self.num_vertices_per_face)
        self.pad_id = pad_id

        self.encoder_attn_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=curr_dim, nhead=8, dropout=attn_dropout)
            for _ in range(attn_encoder_depth)
        ])



    @beartype
    def encode( #Encoder
        self,
        *,
        vertices:         TensorType['b', 'nv', 3, float],
        faces:            TensorType['b', 'nf', 'nvf', int],
        face_edges:       TensorType['b', 'e', 2, int],
        face_mask,
        return_face_coordinates = False,
        in_em
    ):
        device =vertices.device 
        face_coords = jxtget_face_coords(vertices, faces) 
        in_em[3]=transform_to_log_coordinates(in_em[3]) #频率转换为对数坐标 加在encoder里！
        # in_em[:,3]=transform_to_log_coordinates(in_em[:,3]) #频率转换为对数坐标 加在encoder里！
        derived_features , in_em_angle_vec = get_derived_face_featuresjxt(face_coords, in_em, device) #这一步用了2s

        discrete_angle = self.discretize_angle(derived_features['angles'])
        angle_embed = self.angle_embed(discrete_angle)
        discrete_area = self.discretize_area(derived_features['area'])
        area_embed = self.area_embed(discrete_area)
        discrete_normal = self.discretize_normals(derived_features['normals'])
        normal_embed = self.normal_embed(discrete_normal)
        discrete_emnoangle = self.discretize_emnoangle(derived_features['emnoangle']) #jxt
        emnoangle_embed = self.emnoangle_embed(discrete_emnoangle) #jxt
        discrete_emangle = self.discretize_emangle(derived_features['emangle']) #jxt torch.Size([2, 20804, 3])好像有点问题，这dim2的三个值怎么都是一样的
        emangle_embed = self.emangle_embed(discrete_emangle) #jxt torch.Size([2, 20804, 3])
        discrete_emfreq = self.discretize_emfreq(derived_features['emfreq']) #emfreq本来就是离散的 #jxt 2024年5月11日13:36:50我草是不是发现了一个bug，没用对离散法。
        emfreq_embed = self.emfreq_embed(discrete_emfreq)
        discrete_face_coords = self.discretize_face_coords(face_coords) #先把face_coords离散化
        discrete_face_coords = rearrange(discrete_face_coords, 'b nf nv c -> b nf (nv c)') # 9 or 12 coordinates per face #重新排布
        face_coor_embed = self.coor_embed(discrete_face_coords) #在这里把face做成embedding
        face_coor_embed = rearrange(face_coor_embed, 'b nf c d -> b nf (c d)') #再重新排布一下

        face_embed, _ = pack([face_coor_embed, angle_embed, area_embed, normal_embed, emnoangle_embed, emangle_embed, emfreq_embed], 'b nf *') 
        em_embed, _ = pack([emangle_embed, emfreq_embed], 'b nf *') 

        #torch.Size([10, 12996, 1056])
        face_embed = self.project_in(face_embed) 
        #torch.Size([10, 12996, 192])
        face_edges = face_edges.reshape(2, -1)
        orig_face_embed_shape = face_embed.shape[:2]#本来就记下了分批了
        face_embed = face_embed.reshape(-1, face_embed.shape[-1])#torch.Size([129960, 192])
        face_embed = self.init_sage_conv(face_embed, face_edges)#torch.Size([129960, 64])
        face_embed = self.init_encoder_act_and_norm(face_embed)#torch.Size([129960, 64])
        face_embed = face_embed.reshape(orig_face_embed_shape[0], orig_face_embed_shape[1], -1)#回复分批

        in_angle = torch.stack([in_em[1]/180, in_em[2]/360]).t().float().unsqueeze(1).to(device)
        discretized_freq = self.discretize_emfreq2(in_em[3]).to(device).unsqueeze(1)
        # for i,conv in enumerate(self.encoders):
        #     # condfreq=self.condfreqlayers[i](discretized_freq)
        #     # condangle=self.condanglelayers[i](in_angle)
        #     # face_embed = face_embed+condangle+condfreq #自带广播操作
        #     # face_embed = conv(face_embed, face_edges) #为什么变成129960啊。。我草了 不能按照batch么
        #     condfreq = self.condfreqlayers[i](discretized_freq)
        #     condangle = self.condanglelayers[i](in_angle)
        #     face_embed = face_embed + condangle + condfreq  # 自带广播操作
        #     face_embed = face_embed.reshape(-1, face_embed.shape[-1])  # 再次合并批次
        #     face_embed = conv(face_embed, face_edges)  # 图卷积操作
        #     face_embed = face_embed.reshape(orig_face_embed_shape[0], orig_face_embed_shape[1], -1)  # 重新分割批次
        #     '''
        #     torch.Size([129960, 64])
        #     torch.Size([129960, 128])
        #     torch.Size([129960, 256])
        #     torch.Size([129960, 256])
        #     '''
        for i, (conv, act_norm) in enumerate(zip(self.encoders, self.encoder_act_and_norm)):
            condfreq = self.condfreqlayers[i](discretized_freq)
            condangle = self.condanglelayers[i](in_angle)
            face_embed = face_embed + condangle + condfreq  # 自带广播操作
            face_embed = face_embed.reshape(-1, face_embed.shape[-1])  # 再次合并批次
            face_embed = conv(face_embed, face_edges)  # 图卷积操作
            face_embed = act_norm(face_embed)  # 应用激活函数和LayerNorm
            face_embed = face_embed.reshape(orig_face_embed_shape[0], orig_face_embed_shape[1], -1)  # 重新分割批次
            
        shape = (*orig_face_embed_shape, face_embed.shape[-1]) # torch.Size([4, 12996, 576])
        # face_embed = face_embed.new_zeros(shape).masked_scatter(rearrange(face_mask, '... -> ... 1'), face_embed) #多了一层[]而已

        # for linear_attn, attn, ff in self.encoder_attn_blocks: #这一段直接没跑
        #     if exists(linear_attn):
        #         face_embed = linear_attn(face_embed, mask = face_mask) + face_embed

        #     face_embed = attn(face_embed, mask = face_mask) + face_embed
        #     face_embed = ff(face_embed) + face_embed

        for attn_layer in self.encoder_attn_blocks:
            # 确保face_embed的形状适合Transformer层
            # 例如，TransformerEncoderLayer期望输入形状为(seq_len, batch_size, d_model)
            face_embed = face_embed.permute(1, 0, 2)  # (nf, b, d) torch.Size([10, 12996, 576])
            face_embed = attn_layer(face_embed) + face_embed  # (nf, b, d) torch.Size([12996, 10, 576]) 残差骚操作
            # face_embed = attn_layer(face_embed)  # (nf, b, d) torch.Size([12996, 10, 576]) 无残差
            face_embed = face_embed.permute(1, 0, 2)  # (b, nf, d)

        # if not return_face_coordinates:
        #     return face_embed

        return face_embed, discrete_face_coords, em_embed, in_em



    # @numba.jit(nopython=True) 
    @beartype
    def decode( 
        self,
        x, 
        em_embed,
        in_em1,
        device,
    ):
        in_angle = torch.stack([in_em1[1]/180, in_em1[2]/360]).t().float().to(device).unsqueeze(1)
        condangle1 = self.incident_angle_linear1(in_angle)
        condangle2 = self.incident_angle_linear2(in_angle)
        condangle3 = self.incident_angle_linear3(in_angle).reshape(in_angle.shape[0],-1,90,180)
        condangle4 = self.incident_angle_linear4(in_angle).reshape(in_angle.shape[0],-1,180,360)
        condangle5 = self.incident_angle_linear5(in_angle).reshape(in_angle.shape[0],-1,360,720)

        discretized_freq = self.discretize_emfreq(in_em1[3]).to(device).unsqueeze(1)
        condfreq1 = self.emfreq_embed1(discretized_freq)
        condfreq2 = self.emfreq_embed2(discretized_freq)
        condfreq3 = self.emfreq_embed3(discretized_freq).reshape(in_angle.shape[0],-1,90,180)
        condfreq4 = self.emfreq_embed4(discretized_freq).reshape(in_angle.shape[0],-1,180,360)
        condfreq5 = self.emfreq_embed5(discretized_freq).reshape(in_angle.shape[0],-1,360,720)

        # #torch.Size([10, 12996, 576])
        # #torch.Size([10, 12996, 208])
        # x = torch.cat([x, em_embed], dim=2) 
        # #torch.Size([10, 12996, 784])
        pad_size = 22500 - x.size(1)
        x = F.pad(x, (0, 0, 0, pad_size)) 
        x = x.view(x.size(0), -1, 22500) 
        
        # # ------------------------1D Conv+FC-----------------------------
        # torch.Size([10, 784, 22500])
        x = self.conv1d1(x) 
        # torch.Size([10, 64(middim), 2250])
        # x = x + condangle1 
        # x = x + condfreq1

        x = self.fc1d1(x)
        x = x.reshape(x.size(0), -1, 45*90) 
        # torch.Size([10, 64, 4050])
        # x = x + condangle2 
        # x = x + condfreq2
        x = x.reshape(x.size(0), -1, 45, 90) 
        # torch.Size([10, 64, 45, 90])

        # ------------------------2D upConv------------------------------
        x = self.upconv1(x)
        # torch.Size([10, 32, 90, 180])
        x = self.bn1(x)
        x = F.relu(x)
        # x = x + condangle3
        # x = x + condfreq3
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = F.relu(x)
        # x = x + condangle3
        # x = x + condfreq3
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = F.relu(x)
        # x = x + condangle3
        # x = x + condfreq3

        x = self.upconv2(x) 
        #torch.Size([10, 16, 180, 360])
        x = self.bn2(x)
        x = F.relu(x)
        # x = x + condangle4
        # x = x + condfreq4
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = F.relu(x)
        # x = x + condangle4
        # x = x + condfreq4
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = F.relu(x)
        # x = x + condangle4
        # x = x + condfreq4

        x = self.upconv3(x) 
        #torch.Size([10, 8, 361, 721])
        x = self.bn3(x)
        x = F.relu(x)
        # x = x + condangle5
        # x = x + condfreq5
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = F.relu(x)
        # x = x + condangle5
        # x = x + condfreq5
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = F.relu(x)
        # x = x + condangle5
        # x = x + condfreq5

        x = self.conv1x1(x)
        # torch.Size([10, 1, 360, 720])

        # torch.Size([10, 1, 361, 721])
        # x = x[:, :, :, :-1]
        # torch.Size([10, 1, 361, 720])

        return x.squeeze(dim=1)

    # @numba.jit() 
    @beartype
    def forward(
        self,
        *,
        vertices:       TensorType['b', 'nv', 3, float],
        faces:          TensorType['b', 'nf', 'nvf', int],
        face_edges:     Optional[TensorType['b', 'e', 2, int]] = None,
        in_em,
        GT=None,
        logger=None,
        device='cpu',
        lgrcs=False,
        gama=0.001,
        beta=0.0,
        loss_type='L1',
        smooth=False
    ):
        ticc = time.time()
        if not exists(face_edges):
            face_edges = derive_face_edges_from_faces(faces, pad_id = self.pad_id)

        face_mask = reduce(faces != self.pad_id, 'b nf c -> b nf', 'all')

        encoded, __, em_embed, in_em1 = self.encode( 
            vertices = vertices, #顶点
            faces = faces, #面
            face_edges = face_edges, #图论边
            face_mask = face_mask, #面mask
            return_face_coordinates = True,
            in_em = in_em,
        )

        decoded = self.decode(
            encoded, 
            em_embed,
            in_em1,
            device,
        )

        # #平滑后处理：中值滤波+高斯滤波+修改后的smoothloss
        # decoded = decoded.unsqueeze(1)  # 添加 channel 维度
        # decoded = median_filter2d(decoded, kernel_size=5)# 应用中值滤波
        # decoded = gaussian_filter2d(decoded, kernel_size=5, sigma=4, device=device)#两个都用 这个效果好
        # decoded = decoded.squeeze(1)
        if smooth == True:
            decoded = self.smoothing_layer(decoded)

        if GT == None:
            return decoded
        else:
            GT = GT[:,:-1,:] #361*720变360*720
            loss = loss_fn(decoded, GT, loss_type=loss_type, gama=gama)
            # l1loss = nn.L1Loss(reduction='sum')
            # loss = l1loss(decoded,GT)

            # psnr_list = batch_psnr(decoded, GT)
            # ssim_list = batch_ssim(decoded, GT)
            # # mean_psnr = sum(psnr_list) / batch_size
            # mean_psnr = psnr_list.mean()
            # mean_ssim = ssim_list.mean()

            with torch.no_grad():
                psnr_list = psnr(decoded, GT)
                ssim_list = myssim(decoded, GT)
                mse_list = batch_mse(decoded, GT)
                mean_psnr = psnr_list.mean()
                mean_ssim = ssim_list.mean()

                minus = decoded - GT
                mse = ((minus) ** 2).mean()
                nmse = mse / torch.var(GT)
                rmse = torch.sqrt(mse)
                l1 = (decoded-GT).abs().mean()
                percentage_error = (minus / (GT + 1e-4)).abs().mean() * 100

            return loss, decoded, mean_psnr, psnr_list, mean_ssim, ssim_list, mse, nmse, rmse, l1, percentage_error, mse_list

            # return loss, decoded, mean_psnr, psnr_list, mean_ssim, ssim_list, mean_mse