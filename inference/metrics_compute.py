import sys
sys.path.append(".")
sys.path.append("..")
import torch
import torch.nn.functional as F
from pwc import pwc_net
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import imageio.v2 as imageio
from options.metric_options import MetricOptions
import math
from skimage import color
import colour
import pandas


def MSE_LAB(img1, img2):

    img1 = color.rgb2lab(img1)
    img2 = color.rgb2lab(img2)

    de = np.mean(colour.delta_E(img1.astype(np.float32), img2.astype(np.float32),method='CIE 1976'))

    return de

backwarp_tenGrid = {}
backwarp_tenPartial = {}

def estimate(tenFirst, tenSecond, net):
    assert(tenFirst.shape[3] == tenSecond.shape[3])
    assert(tenFirst.shape[2] == tenSecond.shape[2])
    intWidth = tenFirst.shape[3]
    intHeight = tenFirst.shape[2]

    intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
    intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

    tenPreprocessedFirst = F.interpolate(input=tenFirst, 
                            size=(intPreprocessedHeight, intPreprocessedWidth), 
                            mode='bilinear', align_corners=False)
    tenPreprocessedSecond = F.interpolate(input=tenSecond, 
                            size=(intPreprocessedHeight, intPreprocessedWidth), 
                            mode='bilinear', align_corners=False)

    tenFlow = 20.0 * F.interpolate(
                        input=net(tenPreprocessedFirst, tenPreprocessedSecond), 
                        size=(intHeight, intWidth), mode='bilinear', align_corners=False)

    tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
    tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

    return tenFlow[:, :, :, :]

def backwarp(tenInput, tenFlow):
    index = str(tenFlow.shape) + str(tenInput.device)
    if index not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), 
                    tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
        tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), 
                    tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[index] = torch.cat([tenHor, tenVer], 1).to(tenInput.device)

    if index not in backwarp_tenPartial:
        backwarp_tenPartial[index] = tenFlow.new_ones([
                tenFlow.shape[0], 1, tenFlow.shape[2], tenFlow.shape[3]])

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), 
                            tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)
    tenInput = torch.cat([tenInput, backwarp_tenPartial[index]], 1)

    tenOutput = F.grid_sample(input=tenInput, 
                grid=(backwarp_tenGrid[index] + tenFlow).permute(0, 2, 3, 1), 
                mode='bilinear', padding_mode='zeros', align_corners=False)

    return tenOutput

def get_backwarp(tenFirst, tenSecond, net, flow=None):

    if flow is None:
        flow = get_flow(tenFirst, tenSecond, net)
    
    tenoutput = backwarp(tenSecond, flow) 	
    tenMask = tenoutput[:, -1:, :, :]
    tenMask[tenMask > 0.999] = 1.0
    tenMask[tenMask < 1.0] = 0.0
    return tenoutput[:, :-1, :, :] * tenMask, tenMask

def get_backwarp_finetune(self, tenFirst, tenSecond, net, flow=None):
    
		flow = self.get_flow(tenFirst, tenSecond, net)
		tenoutput = self.backwarp(tenSecond, flow)

		tenMask = tenoutput[:, -1:, :, :]
		flow_backward = self.get_flow(tenSecond, tenoutput[:, :-1, :, :], net)
		tenMask_hold = torch.clone(tenMask)
		idx_less = (torch.linalg.norm(flow + flow_backward,dim=1,ord=1)**2) < 0.01*(torch.linalg.norm(flow,dim=1,ord=1)**2 + torch.linalg.norm(flow_backward,dim=1,ord=1)**2) + 0.5

		tenMask[~(torch.unsqueeze(idx_less,1))] = 0.0
		tenMask[torch.unsqueeze(idx_less,1)] = 1.0

		tenMask_hold[tenMask_hold > 0.999] = 1.0
		tenMask_hold[tenMask_hold < 1.0] = 0.0

		final_mask = tenMask_hold*tenMask

		return tenoutput[:, :-1, :, :] * final_mask, final_mask

def get_flow(tenFirst, tenSecond, net):
    with torch.no_grad():
        net.eval()
        flow = estimate(tenFirst, tenSecond, net) 
    return flow


def main(opt):
    """
    Compute metrics for image inference.

    Args:
        opt (argparse.Namespace): Command-line arguments.

    Returns:
        None
    """
    ps = []
    ss = []
    namesl = []
    de_4 = []
    device = "cuda:0"

    pwcnet = pwc_net.PWCNET().to(device)
    names = (pandas.read_csv(opt.meta).to_numpy()[:,0]).tolist()

    for i,n in enumerate(names):

        fake = np.asarray(imageio.imread(opt.path_pred+n[:-4]+'.png'))
        fake = torch.from_numpy(fake.transpose((2, 0, 1)))
        fake = (torch.unsqueeze(fake,0)/255.0).to(device)

        real = np.asarray(imageio.imread(opt.path_gt+n[:-4]+'.png'))
        real = torch.from_numpy(real.transpose((2, 0, 1)))
        real = (torch.unsqueeze(real,0)/255.0).to(device)

        real, mask = get_backwarp(fake,real,pwcnet)

        fake = fake*mask

        real = real.squeeze(0).permute(1,2,0).cpu().numpy()
        fake = fake.squeeze(0).permute(1,2,0).cpu().numpy()

        vpsnr = psnr(real, fake)
        ps.append(vpsnr)
        vssim = ssim(real, fake, data_range=fake.max() - fake.min(), multichannel=True)
        ss.append(vssim)
        delta_E_4 = MSE_LAB(real,fake)
        de_4.append(delta_E_4)
        namesl.append(n)
        print(i)
        print(n,vpsnr,vssim,delta_E_4)

    print("Generating metrics...")
    namesl.append("Average")
    ps.append(sum(ps)/len(ps))
    ss.append(sum(ss)/len(ss))
    de_4.append(sum(de_4)/len(de_4))


    df = pandas.DataFrame(data={"Names": namesl, "PSNR": ps, "SSIM": ss, "DELTA_E":de_4})
    df.to_csv(opt.save_name, sep=',',index=False,float_format='%.6f')
    print("Done!")


if __name__ == '__main__':
    opt = MetricOptions().parse()
    main(opt)



