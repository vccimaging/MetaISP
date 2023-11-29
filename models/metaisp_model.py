import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks as N
from . import losses as L
from util.util import get_coord
from pwc import pwc_net
from functools import partial
import models.attention_models as attention_models

class MetaISPModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt):
        super(MetaISPModel, self).__init__(opt)

        self.opt = opt
        
        if opt.illuminant:
            self.loss_names = ['MetaISPNet_L1', 'MetaISPNet_SSIM', 'MetaISPNet_VGG', 'Illuminant', 'Total']
        else:
            self.loss_names = ['MetaISPNet_L1', 'MetaISPNet_SSIM', 'MetaISPNet_VGG', 'Total'] 
        
        if not opt.latent:
            self.visual_names = ['dslr_warp', 'dslr_mask', 'data_out']
        else:
            self.visual_names = ['data_out']
        self.model_names = ['MetaISPNet']
        self.optimizer_names = ['MetaISPNet_optimizer_%s' % opt.optimizer]
                         
        isp = MetaISPNet(opt)

        self.netMetaISPNet = N.init_net(isp, opt.init_type, opt.init_gain, opt.gpu_ids)


        pwcnet = pwc_net.PWCNET()
        self.netPWCNET = N.init_net(pwcnet, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.set_requires_grad(self.netPWCNET, requires_grad=False)

        if self.isTrain:		
            self.optimizer_MetaISPNet = optim.Adam(self.netMetaISPNet.parameters(),
                                            lr=opt.lr,
                                            betas=(opt.beta1, opt.beta2),
                                            weight_decay=opt.weight_decay)

            self.optimizers = [self.optimizer_MetaISPNet]

            self.criterionL1 = N.init_net(L.L1Loss(), gpu_ids=opt.gpu_ids)
            self.criterionSSIM = N.init_net(L.SSIMLoss(), gpu_ids=opt.gpu_ids)
            self.criterionVGG = N.init_net(L.VGGLoss(), gpu_ids=opt.gpu_ids)
            

        self.data_ispnet_coord = {}

    def set_input(self, input):
        self.iso = input['iso'].to(self.device)
        self.exp = input['exp'].to(self.device)
        self.data_raw = input['raw'].to(self.device)
        self.data_dslr = input['dslr'].to(self.device)
        self.data_raw_demosaic_full = input['raw_demosaic_full'].to(self.device)
        self.wb = input['wb'].to(self.device).float()
        self.image_paths = input['fname']
        self.dslr_image_ref = input['dslr_image_ref'].to(self.device)
        self.coords_atten = input['coords'].to(self.device).float()

        if self.opt.multi_device:
            self.phone = input['device']
            
    def forward(self):
        if not self.opt.latent:
            if self.opt.fine_tune_warp:
                self.dslr_warp, self.dslr_mask = self.get_backwarp_finetune(self.dslr_image_ref, self.data_dslr, self.netPWCNET)
            else:
                self.dslr_warp, self.dslr_mask = self.get_backwarp(self.dslr_image_ref, self.data_dslr, self.netPWCNET)
        
        N, C, H, W = self.data_raw.shape

        index = str(self.data_raw.shape) + '_' + str(self.data_raw.device)
        if index not in self.data_ispnet_coord:
            if self.opt.pre_ispnet_coord:
                data_ispnet_coord = get_coord(H=H, W=W)
            else:
                data_ispnet_coord = get_coord(H=H, W=W, x=1, y=1)
            data_ispnet_coord = np.expand_dims(data_ispnet_coord, axis=0)
            data_ispnet_coord = np.tile(data_ispnet_coord, (N, 1, 1, 1))
            self.data_ispnet_coord[index] = torch.from_numpy(data_ispnet_coord).to(self.data_raw.device)
    
        if self.opt.multi_device:
            if self.opt.illuminant:
                self.data_out, self.illuminant_out = self.netMetaISPNet(self.data_raw, self.data_raw_demosaic_full, self.wb,self.data_ispnet_coord[index],self.phone, self.opt.latent, self.coords_atten,self.exp,self.iso)
            else:
                if self.opt.latent:
                    out = self.netMetaISPNet(self.data_raw, self.data_raw_demosaic_full, self.wb,self.data_ispnet_coord[index],self.phone, self.opt.latent, self.coords_atten,self.exp,self.iso)
                    B,C,H,W = out[0].shape
                    self.data_out = torch.Tensor(2*self.opt.latent_n, C, H, W).to(self.data_raw.device)
                    torch.cat(out, out=self.data_out)
                else:
                    self.data_out = self.netMetaISPNet(self.data_raw, self.data_raw_demosaic_full, self.wb,self.data_ispnet_coord[index],self.phone, self.opt.latent, self.coords_atten,self.exp,self.iso)
        else:
            self.data_out = self.netMetaISPNet(self.data_raw, self.data_raw_demosaic_full, self.wb, self.data_ispnet_coord[index])

        if self.isTrain:
            self.data_out = self.data_out * self.dslr_mask
        else:
            if not self.opt.latent and not self.opt.single_image:
                if self.opt.fine_tune_warp:
                    self.dslr_warp, self.dslr_mask = self.get_backwarp_finetune(self.data_out, self.data_dslr, self.netPWCNET)
                else:
                    self.dslr_warp, self.dslr_mask = self.get_backwarp(self.data_out, self.data_dslr, self.netPWCNET)

    def backward(self):
     
        loss_total = 0

        loss_MetaISPNet_L1 = self.criterionL1(self.data_out, self.dslr_warp).mean()
        loss_MetaISPNet_VGG = self.criterionVGG(self.data_out, self.dslr_warp).mean()
        loss_MetaISPNet_SSIM = (1 - self.criterionSSIM(self.data_out, self.dslr_warp).mean())* 0.15

        if self.opt.illuminant:
            loss_Illuminant = (self.criterionL1(self.illuminant_out, self.wb).mean())*self.opt.wc 
            loss_total =  loss_MetaISPNet_L1 + loss_MetaISPNet_VGG + loss_MetaISPNet_SSIM + loss_Illuminant
        else:
            loss_total =  loss_MetaISPNet_L1 + loss_MetaISPNet_VGG + loss_MetaISPNet_SSIM

        loss_total.backward()
        self.optimizer_MetaISPNet.step()
        self.loss_MetaISPNet_L1 = loss_MetaISPNet_L1.detach()
        self.loss_MetaISPNet_VGG = loss_MetaISPNet_VGG.detach()
        self.loss_MetaISPNet_SSIM = loss_MetaISPNet_SSIM.detach()
        self.loss_Total = loss_total.detach()

        if self.opt.illuminant:
            self.loss_Illuminant = loss_Illuminant.detach()


    def optimize_parameters(self):
        self.optimizer_MetaISPNet.zero_grad()
        self.forward()
        self.backward()
        

class MetaISPNet(nn.Module):
    def __init__(self, opt):
        super(MetaISPNet, self).__init__()

        self.opt = opt

        #Global Feature Extractor
        self.xcit = attention_models.XCiT(patch_size=16, embed_dim=128, depth=4, num_heads=4, mlp_ratio=4, qkv_bias=True,
                            norm_layer=partial(nn.LayerNorm, eps=1e-6), eta=1.0, tokens_norm=False)
       
        #Illutmination Estimation
        if opt.illuminant:
            self.illumination = Illumination()

        #Iso and Exposure mixture and projection
        if self.opt.iso_exp:
            self.IsoExp = IsoExp()

        # White Balance Projection
        self.color_scaler_lv_0 = nn.Conv2d(4, 4, 1, 1, 0)
        self.color_scaler_lv_1 = nn.Conv2d(4, 64, 1, 1, 0)

        ch_1 = 64
        ch_2 = 128
        
        self.multi_device = opt.multi_device
        self.query_emb = opt.query_emb

        if opt.multi_device:
            self.emb = nn.Embedding(3, 128)
            self.emb_lv1 = nn.Conv2d(128, 512, 1, 1, 0, bias=True, padding_mode='zeros')

            if self.opt.illuminant:
                self.emb_4 =  nn.Conv2d(128, 64, 1, 1, 0, bias=True, padding_mode='zeros')

        self.emb_ada_up3 =  nn.Conv2d(128, ch_2, 1, 1, 0, bias=True, padding_mode='zeros') 
        self.emb_ada_up2 =  nn.Conv2d(128, ch_1, 1, 1, 0, bias=True, padding_mode='zeros') 
        self.emb_ada_up1 =  nn.Conv2d(128, ch_1, 1, 1, 0, bias=True, padding_mode='zeros') 
            
        
        self.pre_ispnet_coord = opt.pre_ispnet_coord

        attention_layer = self.opt.attention
        attention_match = self.opt.attention_match

        self.head = N.seq(
            N.conv(4, ch_1, mode='C')
            ) 

        self.down1_0 = N.conv(ch_1+2, ch_1+2, mode='C')
        self.down1_1 = N.RCAGroup(in_channels=ch_1+2, out_channels=ch_1+2,  depth=opt.depth, heads=2, attention=attention_layer,attention_match=attention_match,
                        positional=opt.positional)
        self.down1_2 = N.conv(ch_1+2, ch_1, mode='C')
        self.down1_3 = N.DWTForward(ch_1)


        self.down2_0 = N.conv(ch_1*4, ch_1, mode='C')
        self.down2_1 = N.RCAGroup(in_channels=ch_1, out_channels=ch_1,  depth=opt.depth, heads=4, attention=attention_layer,attention_match=attention_match,
                       positional=opt.positional)
        self.down2_2 = N.DWTForward(ch_1)


        self.down3_0 = N.conv(ch_1*4, ch_2, mode='C')
        self.down3_1 = N.RCAGroup(in_channels=ch_2, out_channels=ch_2,  depth=opt.depth, heads=4, attention='off',attention_match=attention_match,
                        positional=opt.positional)
        self.down3_2 = N.DWTForward(ch_2)


        self.middle_conv1 = N.conv(ch_2*4, ch_2, mode='C')
        self.middle_group1 = N.RCAGroup(in_channels=ch_2, out_channels=ch_2,  depth=opt.depth, heads=4, attention='off',attention_match=attention_match,
                                        positional=opt.positional)
        self.middle_group2 = N.RCAGroup(in_channels=ch_2, out_channels=ch_2,  depth=opt.depth, heads=4, attention='off',attention_match=attention_match,
                                        positional=opt.positional)
        self.middle_conv2 = N.conv(ch_2, ch_2*4, mode='C')
        
        if  self.query_emb:
            self.up3_up = N.DWTInverse(ch_2*4)
            self.up3_att = N.RCAGroup(in_channels=ch_2, out_channels=ch_2,  depth=opt.depth, heads=4, attention='off',attention_match=attention_match,
                                positional=opt.positional, adain=ch_2)
            self.up3_conv = N.conv(ch_2, ch_1*4, mode='C')


            self.up2_up = N.DWTInverse(ch_1*4)
            self.up2_att = N.RCAGroup(in_channels=ch_1, out_channels=ch_1,  depth=opt.depth, heads=4, attention=attention_layer,attention_match=attention_match,
                            positional=opt.positional, adain=ch_1)
            self.up2_conv = N.conv(ch_1, ch_1*4, mode='C')


            self.up1_up = N.DWTInverse(ch_1*4)
            self.up1_att = N.RCAGroup(in_channels=ch_1, out_channels=ch_1,  depth=opt.depth, heads=4, attention=attention_layer,attention_match=attention_match,
                            positional=opt.positional, adain=ch_1)
            self.up1_conv = N.conv(ch_1, ch_1, mode='C')

        self.tail = N.seq(
            N.conv(ch_1, ch_1*4, mode='C'),
            nn.PixelShuffle(upscale_factor=2),
            N.conv(ch_1, 3, mode='C')
            )  # shape: (N, 3, H, W)   

    def forward(self, raw, raw_full, color_scaler=None, coord=None, mobile_id=0, latent=False, coords=None, exp=None, iso=None):
        #Gamma correction
        input = torch.pow(raw, 1/2.2)

        if self.multi_device:
            if latent:
                results = []
                vectors = list()
                embedding_iphone = torch.unsqueeze(torch.unsqueeze(self.emb(torch.tensor([2]).to(raw.device)),2),2)
                embedding_samsung = torch.unsqueeze(torch.unsqueeze(self.emb(torch.tensor([1]).to(raw.device)),2),2)
                embedding_pixel = torch.unsqueeze(torch.unsqueeze(self.emb(torch.tensor([0]).to(raw.device)),2),2)

                ratios = torch.linspace(0, 1, self.opt.latent_n)

                for ratio in ratios:
                    v = (1.0 - ratio) * embedding_iphone + ratio * embedding_pixel
                    vectors.append(v)

                for ratio in ratios:
                    v = (1.0 - ratio) * embedding_pixel + ratio * embedding_samsung
                    vectors.append(v)

                batch_size_inter = self.opt.latent_n
                b = torch.Tensor(batch_size_inter*2, 128, 1, 1).to(raw.device)
                torch.cat(vectors, out=b)
                embedding = b
                
                emb_bottleneck = self.emb_lv1(b)
            else:

                embedding = torch.unsqueeze(torch.unsqueeze(self.emb(mobile_id),2),2)
                emb_bottleneck = self.emb_lv1(embedding)
        
        if self.opt.illuminant:
            color_scaler = self.illumination(input,self.emb_4(embedding))

        features = self.xcit(raw_full)

        color_scaler_0 = self.color_scaler_lv_0(color_scaler)
        color_scaler_64 = self.color_scaler_lv_1(color_scaler_0)

        if latent:
            for i in range(self.opt.latent_n*2):
                if self.opt.iso_exp:
                    raw_p = self.IsoExp(exp.unsqueeze(1).float(),iso.unsqueeze(1).float(),input*color_scaler_0)
                    h = self.head(raw_p)
                else:
                    h = self.head(input*color_scaler_0)

                h_coord = torch.cat((h*color_scaler_64, coord), 1)

                d1 = self.down1_0(h_coord)
                d1 = self.down1_1(d1,coords=coords)
                d1 = self.down1_2(d1)
                d1 = self.down1_3(d1)

                d2 = self.down2_0(d1)
                d2 = self.down2_1(d2,coords=coords)
                d2 = self.down2_2(d2)

                d3 = self.down3_0(d2)
                d3 = self.down3_1(d3,coords=coords)
                d3 = self.down3_2(d3)


                #Bottleneck
                g = self.middle_conv1(d3) 
                g = self.middle_group1(g,coords=coords)
                g = self.middle_group2(g,coords=coords)
                m = (self.middle_conv2(g) + d3)*emb_bottleneck[i].unsqueeze(0)

                #Global Features
                m = m*features

                #Decoding
                if self.query_emb:

                    emb_u3 = self.emb_ada_up3(embedding[i].unsqueeze(0)) 
                    emb_u2 = self.emb_ada_up2(embedding[i].unsqueeze(0))
                    emb_u1 = self.emb_ada_up1(embedding[i].unsqueeze(0))

                    u3 = self.up3_up(m)
                    u3 = self.up3_att(u3,emb_u3,coords=coords)
                    u3 = self.up3_conv(u3) + d2 

                    u2 = self.up2_up(u3)
                    u2 = self.up2_att(u2,emb_u2,coords=coords)
                    u2 = self.up2_conv(u2) + d1

                    u1 = self.up1_up(u2)
                    u1 = self.up1_att(u1,emb_u1,coords=coords)
                    u1 = self.up1_conv(u1) + h  
            
                    out = self.tail(u1)
                else:
                    u3 = self.up3(m) + d2
                    u2 = self.up2(u3) + d1
                    u1 = self.up1(u2) + h
                    out = self.tail(u1)

                results.append(out)

            return results

        else:
            
            if self.opt.iso_exp:
                raw_p = self.IsoExp(exp.unsqueeze(1).float(),iso.unsqueeze(1).float(),input*color_scaler_0)
                h = self.head(raw_p)
            else:
                h = self.head(input*color_scaler_0)

            h_coord = torch.cat((h*color_scaler_64, coord), 1)


            if self.multi_device:

                #Encoding
                d1 = self.down1_0(h_coord)
                #print(coords.shape)
                d1 = self.down1_1(d1,coords=coords)
                d1 = self.down1_2(d1)
                d1 = self.down1_3(d1)

                d2 = self.down2_0(d1)
                d2 = self.down2_1(d2,coords=coords)
                d2 = self.down2_2(d2)

                d3 = self.down3_0(d2)
                d3 = self.down3_1(d3,coords=coords)
                d3 = self.down3_2(d3)


                #Bottleneck
                g = self.middle_conv1(d3) 
                g = self.middle_group1(g,coords=coords)
                g = self.middle_group2(g,coords=coords)
                m = (self.middle_conv2(g) + d3)*emb_bottleneck

                #Global Features
                m = m*features

                #Decoding
                if self.query_emb:

                    emb_u3 = self.emb_ada_up3(embedding) 
                    emb_u2 = self.emb_ada_up2(embedding)
                    emb_u1 = self.emb_ada_up1(embedding)

                    u3 = self.up3_up(m)
                    u3 = self.up3_att(u3,emb_u3,coords=coords)
                    u3 = self.up3_conv(u3) + d2 

                    u2 = self.up2_up(u3)
                    u2 = self.up2_att(u2,emb_u2,coords=coords)
                    u2 = self.up2_conv(u2) + d1

                    u1 = self.up1_up(u2)
                    u1 = self.up1_att(u1,emb_u1,coords=coords)
                    u1 = self.up1_conv(u1) + h  
            
                    out = self.tail(u1)

                else:
                    u3 = self.up3(m) + d2
                    u2 = self.up2(u3) + d1
                    u1 = self.up1(u2) + h
                    out = self.tail(u1)
                
                if  self.opt.illuminant:
                    return out, color_scaler
                else:
                    return out

            else:
                d1 = self.down1(h_coord)
                d2 = self.down2(d1)
                d3 = self.down3(d2)
                m = self.middle_conv1(d3) 
                m = self.middle_group1(m)
                m = self.middle_group2(m)
                m = self.middle_conv2(m) + d3
                #m = self.middle(d3) + d3
                u3 = self.up3(m*features) + d2
                u2 = self.up2(u3) + d1
                u1 = self.up1(u2) + h
                out = self.tail(u1)

                if  self.opt.illuminant:
                    return out, color_scaler
                else:
                    return out
                    
class Illumination(nn.Module):
    def __init__(self):
        super(Illumination, self).__init__()

        self.conv1 = N.conv(4, 64, 7, stride=2, padding=0, mode='CR')
        self.conv2 = N.conv(64, 64, kernel_size=3, stride=1, padding=1, mode='CRC')
        self.adp = nn.AdaptiveMaxPool2d(1)
        self.conv3 = N.conv(64, 4, 1, stride=1, padding=0, mode='C')
		
    def forward(self,raw, embedding):
        raw = self.conv1(raw)
        raw = self.conv2(raw*embedding)
        raw = self.adp(raw)
        raw = self.conv3(raw)
        return raw

class IsoExp(nn.Module):
    def __init__(self):
        super(IsoExp, self).__init__()
        
        self.conv1exp = nn.Linear(1,32)
        self.conv1iso = nn.Linear(1,32)
        self.beta = nn.Linear(64,4)
        self.alpha = nn.Linear(64,4)
        

    def forward(self, exp, iso, img):
        exp = F.relu(self.conv1exp(exp))
        iso = F.relu(self.conv1iso(iso))
        final = torch.cat([exp,iso],dim=1)
        beta = (self.beta(final))
        alpha = (self.alpha(final))
        beta = torch.unsqueeze(torch.unsqueeze(beta,2),2)
        alpha = torch.unsqueeze(torch.unsqueeze(alpha,2),2)
        img = alpha*img + beta
        return img
     