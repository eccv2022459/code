import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import timm

from src.models.video_net import ME_Spynet, GDN, flow_warp
from src.entropy_models.video_entropy_models import BitEstimator
from src.layers.pvt_v2 import pvt_block
from src.layers.Inv_arch import InvNet
from src.layers import pvt_v2
from src.layers.pvt_v2 import Mlp

class Warping_Attention(nn.Module):

    def __init__(self,dim_in,dim_out,qkv_bias=False):

        super().__init__()

        self.q = nn.Linear(dim_in,dim_out,bias=qkv_bias)
        self.k = nn.Linear(dim_in,dim_out,bias=qkv_bias)
        self.v = nn.Linear(dim_in,dim_out,bias=qkv_bias)
        self.v_value = None

    def attention_encoder(self,referfeature,inputfeature):

        B,C,H,W = referfeature.shape
        referfeature = referfeature.view(B,C,H*W).permute(0,2,1).contiguous()
        inputfeature = inputfeature.view(B,C,H*W).permute(0,2,1).contiguous()

        q = self.q(inputfeature)
        k = self.k(referfeature)
        v = self.v(referfeature)
        self.v_value = v

        attn = (q @ k.transpose(-2,-1))
        attn = attn.softmax(dim=-1)
        return attn

    def attention_decoder(self,attn):

        B,N,C = self.v_value.shape
        x = (attn @ self.v_value).transpose(1,2).reshape(B,N,C).permute(0,2,1).contiguous()
        return x

class mvmc_net(nn.Module):

    def __init__(self,args):
        super().__init__()

        self.args = args
        img_height = args.img_height
        img_width = args.img_width

        out_channel_flow = 32
        self.out_channel_flow = out_channel_flow

        out_channel_attn = 16
        self.out_channel_attn = out_channel_attn

        hidden_dim = 64
        self.hidden_dim = hidden_dim

        mc_stage_num = 2
        self.mc_stage_num = mc_stage_num

        inv_block_num = 1
        add_num = 0

        self.opticFlow = ME_Spynet()
        self.warping_attention = Warping_Attention(dim_in=512,dim_out=hidden_dim)

        self.motion_netG = InvNet(channel_in=2,block_num=[inv_block_num,inv_block_num+add_num],down_num=2,img_sizes=[[img_height//8,img_width//8],[img_height//16,img_width//16]],num_heads=[4,8],mlp_ratios=[4,4],sr_ratios=[2,1])
        self.attention_netG = InvNet(channel_in=1,block_num=[inv_block_num,inv_block_num+add_num],down_num=2,img_sizes=[[img_height//8,img_width//8],[img_height//16,img_width//16]],num_heads=[4,8],mlp_ratios=[4,4],sr_ratios=[2,1])
        self.mlp1 = Mlp(in_features=hidden_dim*2,hidden_features=hidden_dim,out_features=hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp2 = Mlp(in_features=hidden_dim,hidden_features=hidden_dim,out_features=hidden_dim)
        self.pvtblock = pvt_block(img_size=[img_height//4,img_width//4],in_chans=hidden_dim,embed_dim=hidden_dim,num_heads=2,mlp_ratio=8,sr_ratio=4)

        self.lrelu = nn.LeakyReLU(inplace=True,negative_slope=0.1)
        img_sizes = [[img_height//4,img_width//4],[img_height//8,img_width//8]]
        num_heads = [2,4]
        mlp_ratios = [8,4]
        sr_ratios = [4,2]
        for i in range(mc_stage_num):
            pvt = pvt_block(img_size=img_sizes[i],in_chans=hidden_dim,embed_dim=hidden_dim,num_heads=num_heads[i],mlp_ratio=mlp_ratios[i],sr_ratio=sr_ratios[i])
            conv = nn.Conv2d(hidden_dim,hidden_dim,3,stride=2,padding=1)
            setattr(self, f"mc_pvt{i}",pvt)
            setattr(self, f"mc_conv{i}",conv)

        img_sizes = [[img_height//8,img_width//8],[img_height//4,img_width//4]]
        num_heads = [4,2]
        mlp_ratios = [4,8]
        sr_ratios = [2,4]
        for i in range(mc_stage_num):
            dpvt = pvt_block(img_size=img_sizes[i],in_chans=hidden_dim,embed_dim=hidden_dim,num_heads=num_heads[i],mlp_ratio=mlp_ratios[i],sr_ratio=sr_ratios[i])
            dconv = nn.ConvTranspose2d(hidden_dim,hidden_dim,3,stride=2,padding=1,output_padding=1)
            setattr(self, f"mc_dpvt{i}",dpvt)
            setattr(self, f"mc_dconv{i}",dconv)

        self.motioncompensation_pred_init = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim,hidden_dim,3,stride=2,padding=1,output_padding=1),
            nn.LeakyReLU(inplace=True,negative_slope=0.1),
            nn.ConvTranspose2d(hidden_dim,3,3,stride=2,padding=1,output_padding=1))

        self.motioncompensation_pred_refine = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim,hidden_dim,3,stride=2,padding=1,output_padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.ConvTranspose2d(hidden_dim,3,3,stride=2,padding=1,output_padding=1))


        self.bitEstimator_z_mv_flow = BitEstimator(out_channel_flow)
        self.bitEstimator_z_mv_attn = BitEstimator(out_channel_attn)

    def motioncompensation(self,mv_warpfeature):

        prediction_init = mv_warpfeature
        prediction_refine = prediction_init

        for i in range(self.mc_stage_num):
            mc_pvt = getattr(self, f"mc_pvt{i}")
            mc_conv = getattr(self, f"mc_conv{i}")
            prediction_refine = mc_pvt(prediction_refine)
            prediction_refine = mc_conv(prediction_refine)
            prediction_refine = self.lrelu(prediction_refine)

        for i in range(self.mc_stage_num):
            mc_dpvt = getattr(self,f"mc_dpvt{i}")
            mc_dconv = getattr(self,f"mc_dconv{i}")
            prediction_refine = mc_dpvt(prediction_refine)
            prediction_refine = mc_dconv(prediction_refine)
            prediction_refine = self.lrelu(prediction_refine)

        prediction_refine_feature = prediction_refine + prediction_init

        prediction_init = self.motioncompensation_pred_init(prediction_init)
        prediction_refine = self.motioncompensation_pred_refine(prediction_refine_feature)
        return prediction_init,prediction_refine,prediction_refine_feature

    def quantize(self, inputs, mode, means=None):
        assert(mode == "dequantize")
        outputs = inputs.clone()
        outputs -= means
        outputs = torch.round(outputs)
        outputs += means
        return outputs

    def iclr18_estrate_bits_z_flow(self, z_mv):
        prob = self.bitEstimator_z_mv_flow(z_mv + 0.5) - self.bitEstimator_z_mv_flow(z_mv - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))
        return total_bits, prob

    def iclr18_estrate_bits_z_attn(self, z_mv):
        prob = self.bitEstimator_z_mv_attn(z_mv + 0.5) - self.bitEstimator_z_mv_attn(z_mv - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))
        return total_bits, prob

    def forward(self,referframe,input_image,referfeatures,inputfeatures,training=True):

        estmv = self.opticFlow(input_image,referframe)

        referfeature_low,referfeature_high = referfeatures
        inputfeature_low,inputfeature_high = inputfeatures

        estmv = F.interpolate(estmv,scale_factor=0.25,mode='nearest')
        mvfeature = self.motion_netG(estmv,rev=False)
        if training:
            noise = torch.nn.init.uniform_(torch.zeros_like(mvfeature),-0.5,0.5)
            quant_flow = mvfeature + noise
        else:
            quant_flow = torch.round(mvfeature)
        quant_mv_decoder = quant_flow
        quant_mv_upsample = self.motion_netG(quant_mv_decoder,rev=True)

        warping_attn = self.warping_attention.attention_encoder(referfeature_high, inputfeature_high)
        warping_attn = warping_attn.unsqueeze(1)
        attnfeature = self.attention_netG(warping_attn,rev=False)
        if training:
            noise = torch.nn.init.uniform_(torch.zeros_like(attnfeature),-0.5,0.5)
            quant_attn = attnfeature + noise
        else:
            quant_attn = torch.round(attnfeature)
        quant_attn_decoder = quant_attn
        quant_attn_upsample = self.attention_netG(quant_attn_decoder,rev=True).squeeze(1)

        mv_flow_warpfeature = flow_warp(referfeature_low,quant_mv_upsample)
        mv_attn_warpfeature = self.warping_attention.attention_decoder(quant_attn_upsample)
        H,W = inputfeature_high.shape[2],inputfeature_high.shape[3]
        B,C,N = mv_attn_warpfeature.shape
        mv_attn_warpfeature = mv_attn_warpfeature.reshape(B,C,H,W)
        mv_attn_warpfeature = F.interpolate(mv_attn_warpfeature,size=mv_flow_warpfeature.size()[2:],mode='bilinear',align_corners=False)

        mv_warpfeature = torch.cat((mv_flow_warpfeature,mv_attn_warpfeature),1)
        B,C,H,W = mv_warpfeature.shape
        mv_warpfeature = mv_warpfeature.reshape(B,C,H*W).permute(0,2,1).contiguous()
        mv_warpfeature = self.mlp1(mv_warpfeature,H,W)
        mv_warpfeature = mv_warpfeature + referfeature_low.reshape(B,C//2,H*W).permute(0,2,1).contiguous()
        mv_warpfeature2 = self.norm2(mv_warpfeature)
        mv_warpfeature2 = self.mlp2(mv_warpfeature2,H,W)
        mv_warpfeature = mv_warpfeature + mv_warpfeature2
        mv_warpfeature = mv_warpfeature.permute(0,2,1).contiguous().reshape(B,C//2,H,W)
        mv_warpfeature = self.pvtblock(mv_warpfeature)

        mv_warpframe,mc_frame,mc_feature = self.motioncompensation(mv_warpfeature)

        total_bits_z_flow, _ = self.iclr18_estrate_bits_z_flow(quant_flow)
        total_bits_z_attn, _ = self.iclr18_estrate_bits_z_attn(quant_attn)
        total_bits_z_mv = total_bits_z_flow + total_bits_z_attn

        im_shape = input_image.size()
        pixel_num = im_shape[0] * im_shape[2] * im_shape[3]
        bpp_mv_z = total_bits_z_mv / pixel_num
        return mv_warpframe,mc_frame,bpp_mv_z,mc_feature

class res_net(nn.Module):

    def __init__(self,args):
        super().__init__()

        self.args = args
        img_height = args.img_height
        img_width = args.img_width

        out_channel_M = 128
        self.out_channel_M = out_channel_M

        out_channel_N = 64
        self.out_channel_N = out_channel_N

        res_stage_num = 4
        self.res_stage_num = res_stage_num

        hidden_dim = 64
        self.hidden_dim = hidden_dim

        inv_block_num = 1
        add_num = 0

        self.res_netG = InvNet(channel_in=hidden_dim,block_num=[inv_block_num,inv_block_num+add_num],down_num=2,img_sizes=[[img_height//8,img_width//8],[img_height//16,img_width//16]],num_heads=[4,8],mlp_ratios=[4,4],sr_ratios=[2,1])
        self.attention_netG = InvNet(channel_in=1,block_num=[inv_block_num,inv_block_num+add_num],down_num=2,img_sizes=[[img_height//8,img_width//8],[img_height//16,img_width//16]],num_heads=[4,8],mlp_ratios=[4,4],sr_ratios=[2,1])
        self.resconv = nn.Conv2d(512,hidden_dim,kernel_size=1,stride=1)
        self.res_attention = Warping_Attention(dim_in=64,dim_out=hidden_dim)

        self.mlp1 = Mlp(in_features=hidden_dim * 2, hidden_features=hidden_dim, out_features=hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp2 = Mlp(in_features=hidden_dim, hidden_features=hidden_dim, out_features=hidden_dim)
        self.pvtblock = pvt_block(img_size=[img_height // 4, img_width // 4], in_chans=hidden_dim, embed_dim=hidden_dim,num_heads=2, mlp_ratio=8, sr_ratio=4)

        self.final_out = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.ConvTranspose2d(hidden_dim, 3, 3, stride=2, padding=1, output_padding=1))

    def forward_final(self,feature):

        output = self.final_out(feature)
        return output

    def forward(self,inputfeatures,mc_feature,training=True):

        inputfeature_low, inputfeature_high = inputfeatures
        referfeature_low, referfeature_high = mc_feature, F.interpolate(mc_feature,size=inputfeature_high.size()[2:],mode='bilinear',align_corners=False)

        res_information = inputfeature_low - referfeature_low
        res_feature = self.res_netG(res_information)
        b, c, h, w = res_feature.size()
        times = c // self.out_channel_M
        res_feature = torch.mean(res_feature.view(b, c // self.out_channel_M, self.out_channel_M, h, w), dim=1)

        feature_renorm = res_feature
        if training:
            noise = torch.nn.init.uniform_(torch.zeros_like(feature_renorm),-0.5,0.5)
            compressed_y_renorm = feature_renorm + noise
        else:
            compressed_y_renorm = torch.round(feature_renorm)
        compressed_y_renorm_decoder = compressed_y_renorm
        compressed_y_renorm_decoder = compressed_y_renorm_decoder.repeat(1,times,1,1)
        recon_res_feature = self.res_netG(compressed_y_renorm_decoder,rev=True)

        inputfeature_high = self.resconv(inputfeature_high)
        res_attn = self.res_attention.attention_encoder(referfeature_high,inputfeature_high)
        res_attn = res_attn.unsqueeze(1)
        attnfeature = self.attention_netG(res_attn,rev=False)
        if training:
            noise = torch.nn.init.uniform_(torch.zeros_like(attnfeature),-0.5,0.5)
            quant_attn = attnfeature + noise
        else:
            quant_attn = torch.round(attnfeature)
        quant_attn_decoder = quant_attn
        quant_attn_upsample = self.attention_netG(quant_attn_decoder,rev=True).squeeze(1)

        res_attn_feature = self.res_attention.attention_decoder(quant_attn_upsample)
        H,W = inputfeature_high.shape[2],inputfeature_high.shape[3]
        B,C,N = res_attn_feature.shape
        res_attn_feature = res_attn_feature.reshape(B,C,H,W)
        res_attn_feature = F.interpolate(res_attn_feature,size=recon_res_feature.size()[2:],mode='bilinear',align_corners=False)

        res_finalfeature = torch.cat((recon_res_feature,res_attn_feature),1)
        B, C, H, W = res_finalfeature.shape
        res_finalfeature = res_finalfeature.reshape(B,C,H*W).permute(0,2,1).contiguous()
        res_finalfeature = self.mlp1(res_finalfeature,H,W)
        res_finalfeature = res_finalfeature + referfeature_low.reshape(B,C//2,H*W).permute(0,2,1).contiguous()
        res_finalfeature2 = self.norm2(res_finalfeature)
        res_finalfeature2 = self.mlp2(res_finalfeature2,H,W)
        res_finalfeature = res_finalfeature + res_finalfeature2
        res_finalfeature= res_finalfeature.permute(0,2,1).contiguous().reshape(B,C//2,H,W)
        res_finalfeature = self.pvtblock(res_finalfeature)

        return res_finalfeature, res_feature, compressed_y_renorm, attnfeature, quant_attn

class res_entroy(nn.Module):

    def __init__(self):

        super().__init__()
        out_channel_M = 128
        self.out_channel_M = out_channel_M

        out_channel_N = 64
        self.out_channel_N = out_channel_N

        attn_out_channel_N = 16
        self.attn_out_channel_N = attn_out_channel_N

        self.priorEncoder = nn.Sequential(
            nn.Conv2d(out_channel_M, out_channel_N, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
        )
        self.priorDecoder = nn.Sequential(
            nn.ConvTranspose2d(out_channel_N, out_channel_M, 5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(out_channel_M, out_channel_M, 5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(out_channel_M, out_channel_M, 3, stride=1, padding=1)
        )
        self.bitEstimator_z = BitEstimator(out_channel_N)

        self.attn_priorEncoder = nn.Sequential(
            nn.Conv2d(16, attn_out_channel_N, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(attn_out_channel_N, attn_out_channel_N, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(attn_out_channel_N, attn_out_channel_N, 5, stride=2, padding=2),
        )
        self.attn_priorDecoder = nn.Sequential(
            nn.ConvTranspose2d(attn_out_channel_N, 16, 5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(16, 16, 5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(16, 16, 3, stride=1, padding=1)
        )
        self.attn_bitEstimator_z = BitEstimator(attn_out_channel_N)

    def iclr18_estrate_bits_z(self, z):
        prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))
        return total_bits,prob

    def iclr18_estrate_bits_z_attn(self, z):
        prob = self.attn_bitEstimator_z(z + 0.5) - self.attn_bitEstimator_z(z - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))
        return total_bits,prob

    def feature_probs_based_sigma(self,feature,sigma):
        mu = torch.zeros_like(sigma)
        maxx = 1e10 if sigma.dtype == torch.float32 else float("inf")
        sigma = sigma.clamp(1e-5,maxx)
        gaussian = torch.distributions.laplace.Laplace(mu, sigma)
        probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50))
        return total_bits, probs

    def res_forward(self,res_feature,compressed_y_renorm,training=True):

        z = self.priorEncoder(res_feature)
        if training:
            noise = torch.nn.init.uniform_(torch.zeros_like(z),-0.5,0.5)
            compressed_z = z + noise
        else:
            compressed_z = torch.round(z)
        recon_sigma = self.priorDecoder(compressed_z)

        total_bits_y, _ = self.feature_probs_based_sigma(compressed_y_renorm,recon_sigma)
        total_bits_z, _ = self.iclr18_estrate_bits_z(compressed_z)
        total_bits = total_bits_y + total_bits_z
        return total_bits

    def attn_forward(self, attnfeature, quant_attn,training=True):

        z = self.attn_priorEncoder(attnfeature)
        if training:
            noise = torch.nn.init.uniform_(torch.zeros_like(z),-0.5,0.5)
            compressed_z = z + noise
        else:
            compressed_z = torch.round(z)
        recon_sigma = self.attn_priorDecoder(compressed_z)

        total_bits_y, _ = self.feature_probs_based_sigma(quant_attn,recon_sigma)
        total_bits_z, _ = self.iclr18_estrate_bits_z_attn(compressed_z)
        total_bits = total_bits_y + total_bits_z
        return total_bits

class InvTrans_net(nn.Module):

    def __init__(self,mvmc_need_grad=True,res_need_grad=True,res_entroy_need_grad=True,args=None):
        super().__init__()

        self.args = args
        img_height = args.img_height
        img_width = args.img_width
        self.featureextract = timm.create_model('pvt_v2_b2_li',pretrained=False, num_stages=4,img_size=[img_height,img_width])
        checkpoint = torch.load("featureextract.pth",map_location='cpu')
        self.featureextract.load_state_dict(checkpoint)

        self.mvmc = mvmc_net(args=args)
        checkpoint = torch.load("mvmc.pth",map_location='cpu')
        self.mvmc.load_state_dict(checkpoint)
        if mvmc_need_grad == False:
            for para in self.mvmc.parameters():
                para.requires_grad = False

        self.res = res_net(args=args)
        checkpoint = torch.load("res.pth",map_location='cpu')
        self.res.load_state_dict(checkpoint)
        if res_need_grad == False:
            for para in self.res.parameters():
                para.requires_grad = False

        self.res_entroy = res_entroy()
        if res_entroy_need_grad == False:
            for para in self.res_entroy.parameters():
                para.requires_grad = False

    def forward(self,referframe,input_image,training):

        referfeatures = self.featureextract(referframe)
        referfeature_low,referfeature_high = referfeatures[0],referfeatures[3]
        referfeatures = [referfeature_low,referfeature_high]

        inputfeatures = self.featureextract(input_image)
        inputfeature_low,inputfeature_high = inputfeatures[0],inputfeatures[3]
        inputfeatures = [inputfeature_low,inputfeature_high]

        mv_warpframe,mc_frame,bpp_mv_z,mc_feature = self.mvmc(referframe,input_image,referfeatures,inputfeatures,training=training)

        res_finalfeature, res_feature, compressed_y_renorm, attnfeature, quant_attn = self.res(inputfeatures,mc_feature,training=training)

        res_total_bits = self.res_entroy.res_forward(res_feature,compressed_y_renorm,training=training)
        attn_total_bits = self.res_entroy.attn_forward(attnfeature,quant_attn,training=training)
        total_bits = res_total_bits + attn_total_bits

        res_finalfeature = res_finalfeature + mc_feature
        recon_image_final =  self.res.forward_final(res_finalfeature)

        im_shape = input_image.size()
        pixel_num = im_shape[0] * im_shape[2] * im_shape[3]
        bpp_res = total_bits / pixel_num

        return recon_image_final,bpp_res,bpp_mv_z,mv_warpframe,mc_frame

    def update(self,force=False):
        self.bitEstimator_z_mv.update(force=force)
        self.bitEstimator_z.update(force=force)
        self.gaussian_encoder.update(force=force)

    def load_dict(self, pretrained_dict):
        result_dict = {}
        for key, weight in pretrained_dict.items():
            result_key = key
            if key[:7] == "module.":
                result_key = key[7:]
            result_dict[result_key] = weight

        self.load_state_dict(result_dict)

def _video_coding_solver(args):

    net = InvTrans_net(mvmc_need_grad=False,res_entroy_need_grad=False,res_need_grad=False,args=args)
    return net
