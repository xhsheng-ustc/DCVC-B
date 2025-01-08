import time

import torch
from torch import nn
import numpy as np

from .common_model import CompressionModel
from .video_net import ME_Spynet, ResBlock, UNet, bilinearupsacling, bilineardownsacling, \
    get_hyper_enc_dec_models, flow_warp
from ..layers.layers import subpel_conv3x3, subpel_conv1x1, DepthConvBlock, \
    ResidualBlockWithStride, ResidualBlockUpsample
from ..utils.stream_helper import get_downsampled_shape, encode_p, decode_p, filesize, \
    get_state_dict

g_ch_1x = 48
g_ch_2x = 64
g_ch_4x = 96
g_ch_8x = 96
g_ch_16x = 128


class OffsetDiversity(nn.Module):
    def __init__(self, in_channel=g_ch_1x, aux_feature_num=g_ch_1x*2+3*4+2*4,
                 offset_num=2, group_num=16, max_residue_magnitude=40, inplace=False):
        super().__init__()
        self.in_channel = in_channel
        self.offset_num = offset_num
        self.group_num = group_num
        self.max_residue_magnitude = max_residue_magnitude
        self.conv_offset = nn.Sequential(
            nn.Conv2d(aux_feature_num, g_ch_2x, 3, 2, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=inplace),
            nn.Conv2d(g_ch_2x, g_ch_2x, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=inplace),
        )
        self.conv_offset_l = nn.Conv2d(g_ch_2x, 3 * group_num * offset_num, 3, 1, 1)
        self.conv_offset_r = nn.Conv2d(g_ch_2x, 3 * group_num * offset_num, 3, 1, 1)
        self.fusion_l = nn.Conv2d(in_channel * offset_num, in_channel, 1, 1, groups=group_num)
        self.fusion_r = nn.Conv2d(in_channel * offset_num, in_channel, 1, 1, groups=group_num)


    def forward(self, x_l, x_r, aux_feature, flow_l, flow_r):
        B, C, H, W = x_l.shape
        out = self.conv_offset(aux_feature)
        out_l = self.conv_offset_l(out)
        out_r = self.conv_offset_l(out)
        
        out_l = bilinearupsacling(out_l)
        out_r = bilinearupsacling(out_r)
        o1_l, o2_l, mask_l = torch.chunk(out_l, 3, dim=1)
        o1_r, o2_r, mask_r = torch.chunk(out_r, 3, dim=1)
        mask_l = torch.sigmoid(mask_l)
        mask_r = torch.sigmoid(mask_r)
        # offset
        offset_l = self.max_residue_magnitude * torch.tanh(torch.cat((o1_l, o2_l), dim=1))
        offset_r = self.max_residue_magnitude * torch.tanh(torch.cat((o1_r, o2_r), dim=1))
        offset_l = offset_l + flow_l.repeat(1, self.group_num * self.offset_num, 1, 1)
        offset_r = offset_r + flow_r.repeat(1, self.group_num * self.offset_num, 1, 1)

        # warp
        offset_l = offset_l.view(B * self.group_num * self.offset_num, 2, H, W)
        offset_r = offset_r.view(B * self.group_num * self.offset_num, 2, H, W)
        mask_l = mask_l.view(B * self.group_num * self.offset_num, 1, H, W)
        mask_r = mask_r.view(B * self.group_num * self.offset_num, 1, H, W)
        x_l = x_l.view(B * self.group_num, C // self.group_num, H, W)
        x_l = x_l.repeat(self.offset_num, 1, 1, 1)
        x_l = flow_warp(x_l, offset_l)
        x_l = x_l * mask_l
        x_l = x_l.view(B, C * self.offset_num, H, W)
        x_l = self.fusion_l(x_l)

        x_r = x_r.view(B * self.group_num, C // self.group_num, H, W)
        x_r = x_r.repeat(self.offset_num, 1, 1, 1)
        x_r = flow_warp(x_r, offset_r)
        x_r = x_r * mask_r
        x_r = x_r.view(B, C * self.offset_num, H, W)
        x_r = self.fusion_r(x_r)

        return x_l, x_r


class FeatureExtractor(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.conv1 = nn.Conv2d(g_ch_1x, g_ch_1x, 3, stride=1, padding=1)
        self.res_block1 = ResBlock(g_ch_1x, inplace=inplace)
        self.conv2 = nn.Conv2d(g_ch_1x, g_ch_2x, 3, stride=2, padding=1)
        self.res_block2 = ResBlock(g_ch_2x, inplace=inplace)
        self.conv3 = nn.Conv2d(g_ch_2x, g_ch_4x, 3, stride=2, padding=1)
        self.res_block3 = ResBlock(g_ch_4x, inplace=inplace)

    def forward(self, feature):
        layer1 = self.conv1(feature)
        layer1 = self.res_block1(layer1)

        layer2 = self.conv2(layer1)
        layer2 = self.res_block2(layer2)

        layer3 = self.conv3(layer2)
        layer3 = self.res_block3(layer3)

        return layer1, layer2, layer3


class MultiScaleContextFusion(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.conv3_up = subpel_conv3x3(g_ch_4x, g_ch_2x, 2)
        self.res_block3_up = ResBlock(g_ch_2x, inplace=inplace)
        self.conv3_out = nn.Conv2d(g_ch_4x, g_ch_4x, 3, padding=1)
        self.res_block3_out = ResBlock(g_ch_4x, inplace=inplace)
        self.conv2_up = subpel_conv3x3(g_ch_2x * 2, g_ch_1x, 2)
        self.res_block2_up = ResBlock(g_ch_1x, inplace=inplace)
        self.conv2_out = nn.Conv2d(g_ch_2x * 2, g_ch_2x, 3, padding=1)
        self.res_block2_out = ResBlock(g_ch_2x, inplace=inplace)
        self.conv1_out = nn.Conv2d(g_ch_1x * 2, g_ch_1x, 3, padding=1)
        self.res_block1_out = ResBlock(g_ch_1x, inplace=inplace)

    def forward(self, context1, context2, context3):
        context3_up = self.conv3_up(context3)
        context3_up = self.res_block3_up(context3_up)
        context3_out = self.conv3_out(context3)
        context3_out = self.res_block3_out(context3_out)
        context2_up = self.conv2_up(torch.cat((context3_up, context2), dim=1))
        context2_up = self.res_block2_up(context2_up)
        context2_out = self.conv2_out(torch.cat((context3_up, context2), dim=1))
        context2_out = self.res_block2_out(context2_out)
        context1_out = self.conv1_out(torch.cat((context2_up, context1), dim=1))
        context1_out = self.res_block1_out(context1_out)
        context1 = context1 + context1_out
        context2 = context2 + context2_out
        context3 = context3 + context3_out

        return context1, context2, context3


class MvEnc(nn.Module):
    def __init__(self, input_channel, channel, inplace=False):
        super().__init__()
        self.enc_1 = nn.Sequential(
            ResidualBlockWithStride(input_channel, channel, stride=2, inplace=inplace),
            DepthConvBlock(channel, channel, inplace=inplace),
        )
        self.enc_2 = ResidualBlockWithStride(channel, channel, stride=2, inplace=inplace)

        self.adaptor_0 = DepthConvBlock(channel, channel, inplace=inplace)
        self.adaptor_1 = DepthConvBlock(channel * 3, channel, inplace=inplace)
        self.adaptor_2 = DepthConvBlock(channel * 3, channel, inplace=inplace)
        self.adaptor_3 = DepthConvBlock(channel * 5, channel, inplace=inplace)
        self.enc_3 = nn.Sequential(
            ResidualBlockWithStride(channel, channel, stride=2, inplace=inplace),
            DepthConvBlock(channel, channel, inplace=inplace),
            nn.Conv2d(channel, channel, 3, stride=2, padding=1),
        )

    def forward(self, est_mv_x_l, est_mv_x_r, ref_mv_l_r, ref_mv_r_l, context_l, context_r, quant_step):
        out = self.enc_1(torch.cat((est_mv_x_l-ref_mv_r_l/2,est_mv_x_r-ref_mv_l_r/2),dim=1))
        out = out * quant_step
        if context_l is None and context_r is None:
            out = self.adaptor_0(out)
        elif context_l is None and context_r is not None:
            out = self.adaptor_1(torch.cat((out, context_r), dim=1))
        elif context_l is not None and context_r is None:
            out = self.adaptor_2(torch.cat((out, context_l), dim=1))
        elif context_l is not None and context_r is None:
            out = self.adaptor_3(torch.cat((out, context_l, context_r), dim=1))
        out = self.enc_2(out)
        return self.enc_3(out)


class MvDec(nn.Module):
    def __init__(self, output_channel, channel, inplace=False):
        super().__init__()
        self.dec_1 = nn.Sequential(
            DepthConvBlock(channel, channel, inplace=inplace),
            ResidualBlockUpsample(channel, channel, 2, inplace=inplace),
            DepthConvBlock(channel, channel, inplace=inplace),
            ResidualBlockUpsample(channel, channel, 2, inplace=inplace),
            DepthConvBlock(channel, channel, inplace=inplace)
        )
        self.dec_2 = ResidualBlockUpsample(channel, channel, 2, inplace=inplace)
        self.dec_3 = nn.Sequential(
            DepthConvBlock(channel, channel, inplace=inplace),
        )
        self.out_l_1 = DepthConvBlock(channel, channel, inplace=inplace)
        self.out_l_2 = subpel_conv1x1(channel, output_channel, 2)
        self.out_r_1 = DepthConvBlock(channel, channel, inplace=inplace)
        self.out_r_2 = subpel_conv1x1(channel, output_channel, 2)

    def forward(self, x, ref_mv_l_r, ref_mv_r_l, quant_step):
        feature = self.dec_1(x)
        out = self.dec_2(feature)
        out = out * quant_step
        out = self.dec_3(out)
        fea_l = self.out_l_1(out)
        hat_mv_x_l = self.out_l_2(fea_l)+ref_mv_r_l/2
        fea_r = self.out_r_1(out)
        hat_mv_x_r = self.out_r_2(fea_r)+ref_mv_l_r/2
        return hat_mv_x_l, hat_mv_x_r, torch.cat((fea_l, fea_r),dim=1)


class ContextualEncoder(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.conv1 = nn.Conv2d(g_ch_1x*2 + 3, g_ch_2x, 3, stride=2, padding=1)
        self.res1 = ResBlock(g_ch_2x * 3, bottleneck=True, slope=0.1,
                             end_with_relu=True, inplace=inplace)
        self.conv2 = nn.Conv2d(g_ch_2x * 3, g_ch_4x, 3, stride=2, padding=1)
        self.res2 = ResBlock(g_ch_4x * 3, bottleneck=True, slope=0.1,
                             end_with_relu=True, inplace=inplace)
        self.conv3 = nn.Conv2d(g_ch_4x * 3, g_ch_8x, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(g_ch_8x, g_ch_16x, 3, stride=2, padding=1)

    def forward(self, x, context1_l, context2_l, context3_l, context1_r, context2_r, context3_r, quant_step):
        feature = self.conv1(torch.cat([x, context1_l, context1_r], dim=1))
        feature = self.res1(torch.cat([feature, context2_l, context2_r], dim=1))
        feature = feature * quant_step
        feature = self.conv2(feature)
        feature = self.res2(torch.cat([feature, context3_l, context3_r], dim=1))
        feature = self.conv3(feature)
        feature = self.conv4(feature)
        return feature


class ContextualDecoder(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.up1 = subpel_conv3x3(g_ch_16x, g_ch_8x, 2)
        self.up2 = subpel_conv3x3(g_ch_8x, g_ch_4x, 2)
        self.res1 = ResBlock(g_ch_4x * 3, bottleneck=True, slope=0.1,
                             end_with_relu=True, inplace=inplace)
        self.up3 = subpel_conv3x3(g_ch_4x * 3, g_ch_2x, 2)
        self.res2 = ResBlock(g_ch_2x * 3, bottleneck=True, slope=0.1,
                             end_with_relu=True, inplace=inplace)
        self.up4 = subpel_conv3x3(g_ch_2x * 3, 32, 2)
        self.out_conv = nn.Conv2d(32+g_ch_1x*2, g_ch_1x, 3, stride=1, padding=1)

    def forward(self, x, context1_l, context2_l, context3_l, context1_r, context2_r, context3_r, quant_step):
        feature = self.up1(x)
        feature = self.up2(feature)
        feature = self.res1(torch.cat([feature, context3_l, context3_r], dim=1))
        feature = self.up3(feature)
        feature = feature * quant_step
        feature = self.res2(torch.cat([feature, context2_l, context2_r], dim=1))
        feature = self.up4(feature)
        feature = self.out_conv(torch.cat([feature, context1_l, context1_r], dim=1))
        return feature

class ReconGeneration(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.unet_1 = UNet(g_ch_1x, g_ch_1x, inplace=inplace)
        self.unet_2 = UNet(g_ch_1x, g_ch_1x, inplace=inplace)
        self.recon_conv = nn.Conv2d(g_ch_1x, 3, 3, stride=1, padding=1)

    def forward(self, res):
        feature = self.unet_1(res)
        feature = self.unet_2(feature)
        recon = self.recon_conv(feature)
        return feature, recon


class DMC(CompressionModel):
    def __init__(self, anchor_num=4, ec_thread=False, stream_part=1, inplace=False):
        super().__init__(y_distribution='laplace', z_channel=g_ch_16x, mv_z_channel=64,
                         ec_thread=ec_thread, stream_part=stream_part)

        channel_mv = 64
        channel_N = 64

        self.optic_flow = ME_Spynet()
        self.align = OffsetDiversity(inplace=inplace)

        self.mv_encoder = MvEnc(4, channel_mv)
        self.mv_hyper_prior_encoder, self.mv_hyper_prior_decoder = \
            get_hyper_enc_dec_models(channel_mv, channel_N, inplace=inplace)

        self.mv_y_prior_fusion = nn.Sequential(
            DepthConvBlock(channel_mv * 2, channel_mv * 3, inplace=inplace),
            DepthConvBlock(channel_mv * 3, channel_mv * 3, inplace=inplace),
        )

        self.mv_y_spatial_prior_adaptor_1 = nn.Conv2d(channel_mv * 4, channel_mv * 3, 1)
        self.mv_y_spatial_prior_adaptor_2 = nn.Conv2d(channel_mv * 4, channel_mv * 3, 1)
        self.mv_y_spatial_prior_adaptor_3 = nn.Conv2d(channel_mv * 4, channel_mv * 3, 1)

        self.mv_y_spatial_prior = nn.Sequential(
            DepthConvBlock(channel_mv * 3, channel_mv * 3, inplace=inplace),
            DepthConvBlock(channel_mv * 3, channel_mv * 3, inplace=inplace),
            DepthConvBlock(channel_mv * 3, channel_mv * 2, inplace=inplace),
        )

        self.mv_decoder = MvDec(2, channel_mv, inplace=inplace)
        self.r_l_mask = nn.Sequential(
            DepthConvBlock(3*4+2*4, channel_mv, inplace=inplace),
            DepthConvBlock(channel_mv, channel_mv, inplace=inplace),
            nn.Conv2d(channel_mv, 1, 3, 1, 1)
        )

        self.feature_adaptor_I = nn.Conv2d(3, g_ch_1x, 3, stride=1, padding=1)
        self.feature_adaptor = nn.ModuleList([nn.Conv2d(g_ch_1x, g_ch_1x, 1) for _ in range(5)])
        self.feature_extractor = FeatureExtractor(inplace=inplace)
        self.context_fusion_net_l = MultiScaleContextFusion(inplace=inplace)
        self.context_fusion_net_r = MultiScaleContextFusion(inplace=inplace)
        self.contextual_encoder = ContextualEncoder(inplace=inplace)

        self.contextual_hyper_prior_encoder, self.contextual_hyper_prior_decoder = \
            get_hyper_enc_dec_models(g_ch_16x, g_ch_16x, True, inplace=inplace)

        self.mv_temporal_prior_encoder = nn.Sequential(
            nn.Conv2d(2*2, g_ch_1x, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=inplace),
            nn.Conv2d(g_ch_1x, channel_mv, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=inplace),
            nn.Conv2d(channel_mv, channel_mv, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=inplace),
            nn.Conv2d(channel_mv, channel_mv, 3, stride=2, padding=1),
        )

        self.temporal_prior_encoder = nn.Sequential(
            nn.Conv2d(g_ch_4x*2, g_ch_8x, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=inplace),
            nn.Conv2d(g_ch_8x, g_ch_16x, 3, stride=2, padding=1),
        )

        self.y_prior_fusion_adaptor_0 = DepthConvBlock(g_ch_16x * 2, g_ch_16x * 3,
                                                       inplace=inplace)
        self.y_prior_fusion_adaptor_1 = DepthConvBlock(g_ch_16x * 3, g_ch_16x * 3,
                                                       inplace=inplace)
        self.y_prior_fusion_adaptor_2 = DepthConvBlock(g_ch_16x * 4, g_ch_16x * 3,
                                                       inplace=inplace)
        self.y_prior_fusion = nn.Sequential(
            DepthConvBlock(g_ch_16x * 3, g_ch_16x * 3, inplace=inplace),
            DepthConvBlock(g_ch_16x * 3, g_ch_16x * 3, inplace=inplace),
        )

        self.y_spatial_prior_adaptor_1 = nn.Conv2d(g_ch_16x * 4, g_ch_16x * 3, 1)
        self.y_spatial_prior_adaptor_2 = nn.Conv2d(g_ch_16x * 4, g_ch_16x * 3, 1)
        self.y_spatial_prior_adaptor_3 = nn.Conv2d(g_ch_16x * 4, g_ch_16x * 3, 1)

        self.y_spatial_prior = nn.Sequential(
            DepthConvBlock(g_ch_16x * 3, g_ch_16x * 3, inplace=inplace),
            DepthConvBlock(g_ch_16x * 3, g_ch_16x * 3, inplace=inplace),
            DepthConvBlock(g_ch_16x * 3, g_ch_16x * 2, inplace=inplace),
        )

        self.contextual_decoder = ContextualDecoder(inplace=inplace)
        self.recon_generation_net = ReconGeneration(inplace=inplace)
        self.mv_y_q_basic_enc = nn.Parameter(torch.ones((1, channel_mv, 1, 1)))
        self.mv_y_q_scale_enc = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.mv_y_q_scale_enc_fine = None
        self.mv_y_q_basic_dec = nn.Parameter(torch.ones((1, channel_mv, 1, 1)))
        self.mv_y_q_scale_dec = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.mv_y_q_scale_dec_fine = None

        self.y_q_basic_enc = nn.Parameter(torch.ones((1, g_ch_2x * 3, 1, 1)))
        self.y_q_scale_enc = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.y_q_scale_enc_fine = None
        self.y_q_basic_dec = nn.Parameter(torch.ones((1, g_ch_2x, 1, 1)))
        self.y_q_scale_dec = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.y_q_scale_dec_fine = None
        self.anchor_num = int(anchor_num)
        self.dpb_list = []

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict)

        with torch.no_grad():
            mv_y_q_scale_enc_fine = np.linspace(np.log(self.mv_y_q_scale_enc[0, 0, 0, 0]),
                                                np.log(self.mv_y_q_scale_enc[3, 0, 0, 0]), 64)
            self.mv_y_q_scale_enc_fine = np.exp(mv_y_q_scale_enc_fine)
            mv_y_q_scale_dec_fine = np.linspace(np.log(self.mv_y_q_scale_dec[0, 0, 0, 0]),
                                                np.log(self.mv_y_q_scale_dec[3, 0, 0, 0]), 64)
            self.mv_y_q_scale_dec_fine = np.exp(mv_y_q_scale_dec_fine)

            y_q_scale_enc_fine = np.linspace(np.log(self.y_q_scale_enc[0, 0, 0, 0]),
                                             np.log(self.y_q_scale_enc[3, 0, 0, 0]), 64)
            self.y_q_scale_enc_fine = np.exp(y_q_scale_enc_fine)
            y_q_scale_dec_fine = np.linspace(np.log(self.y_q_scale_dec[0, 0, 0, 0]),
                                             np.log(self.y_q_scale_dec[3, 0, 0, 0]), 64)
            self.y_q_scale_dec_fine = np.exp(y_q_scale_dec_fine)

    def multi_scale_feature_extractor(self, ref_feature, ref_frame, layer_id):
        if ref_feature is None:
            feature = self.feature_adaptor_I(ref_frame)
        else:
            feature = self.feature_adaptor[layer_id](ref_feature)
        fea1, fea2, fea3 = self.feature_extractor(feature)
        return fea1, fea2, fea3

    def motion_compensation(self, dpb_l, dpb_r, ref_x_l, ref_x_r, mv_hat_x_l, mv_hat_x_r, ref_mv_l_r, ref_mv_r_l, layer_id):
        warpframe_l = flow_warp(ref_x_l, mv_hat_x_l)
        ref_feature1_l, ref_feature2_l, ref_feature3_l = self.multi_scale_feature_extractor(dpb_l["ref_feature"], dpb_l["ref_frame"], layer_id)

        warpframe_r = flow_warp(ref_x_r, mv_hat_x_r)
        ref_feature1_r, ref_feature2_r, ref_feature3_r = self.multi_scale_feature_extractor(dpb_r["ref_feature"], dpb_r["ref_frame"], layer_id)

        mv2_x_l = bilineardownsacling(mv_hat_x_l) / 2
        mv3_x_l = bilineardownsacling(mv2_x_l) / 2

        mv2_x_r = bilineardownsacling(mv_hat_x_r) / 2
        mv3_x_r = bilineardownsacling(mv2_x_r) / 2
        context1_l_init = flow_warp(ref_feature1_l, mv_hat_x_l)
        context1_r_init = flow_warp(ref_feature1_r, mv_hat_x_r)


        context1_l, context1_r = self.align(ref_feature1_l, ref_feature1_r, torch.cat(
            (context1_l_init, context1_r_init, warpframe_l, warpframe_r, ref_x_l, ref_x_r, \
             mv_hat_x_l, mv_hat_x_r, ref_mv_r_l/2, ref_mv_l_r/2), dim=1), mv_hat_x_l, mv_hat_x_r)
        
        context2_l = flow_warp(ref_feature2_l, mv2_x_l)
        context3_l = flow_warp(ref_feature3_l, mv3_x_l)

        context2_r = flow_warp(ref_feature2_r, mv2_x_r)
        context3_r = flow_warp(ref_feature3_r, mv3_x_r)

        context1_l, context2_l, context3_l = self.context_fusion_net_l(context1_l, context2_l, context3_l)
        context1_r, context2_r, context3_r = self.context_fusion_net_r(context1_r, context2_r, context3_r)
        return context1_l, context2_l, context3_l, context1_r, context2_r, context3_r

    @staticmethod
    def get_q_scales_from_ckpt(ckpt_path):
        ckpt = get_state_dict(ckpt_path)
        y_q_scale_enc = ckpt["y_q_scale_enc"].reshape(-1)
        y_q_scale_dec = ckpt["y_q_scale_dec"].reshape(-1)
        y_q_basic_enc = ckpt["y_q_basic_enc"].reshape(-1)
        y_q_basic_dec = ckpt["y_q_basic_dec"].reshape(-1)
        mv_y_q_scale_enc = ckpt["mv_y_q_scale_enc"].reshape(-1)
        mv_y_q_scale_dec = ckpt["mv_y_q_scale_dec"].reshape(-1)
        mv_y_q_basic_enc = ckpt["mv_y_q_basic_enc"].reshape(-1)
        mv_y_q_basic_dec = ckpt["mv_y_q_basic_dec"].reshape(-1)
        return y_q_basic_enc, y_q_basic_dec, y_q_scale_enc, y_q_scale_dec, mv_y_q_basic_enc, mv_y_q_basic_dec, mv_y_q_scale_enc, mv_y_q_scale_dec

    def mv_prior_param_decoder(self, mv_z_hat, ref_mv_l_r, ref_mv_r_l, slice_shape=None):
        mv_params = self.mv_hyper_prior_decoder(mv_z_hat)
        mv_params = self.slice_to_y(mv_params, slice_shape)
        mv_temporal_params = self.mv_temporal_prior_encoder(torch.cat((ref_mv_l_r, ref_mv_r_l), dim=1))
        mv_params = torch.cat((mv_params, mv_temporal_params), dim=1)
        mv_params = self.mv_y_prior_fusion(mv_params)
        return mv_params

    def res_prior_param_decoder(self, z_hat, dpb_l, dpb_r, context3_l, context3_r, slice_shape=None):
        hierarchical_params = self.contextual_hyper_prior_decoder(z_hat)
        hierarchical_params = self.slice_to_y(hierarchical_params, slice_shape)
        temporal_params = self.temporal_prior_encoder(torch.cat((context3_l, context3_r), dim=1))
        ref_y_l = dpb_l["ref_y"]
        ref_y_r = dpb_r["ref_y"]
        if ref_y_l is None and ref_y_r is None:
            params = torch.cat((temporal_params, hierarchical_params), dim=1)
            params = self.y_prior_fusion_adaptor_0(params)
        elif ref_y_l is None and ref_y_r is not None:
            params = torch.cat((temporal_params, hierarchical_params, ref_y_r), dim=1)
            params = self.y_prior_fusion_adaptor_1(params)
        elif ref_y_l is not None and ref_y_r is None:
            params = torch.cat((temporal_params, hierarchical_params, ref_y_l), dim=1)
            params = self.y_prior_fusion_adaptor_1(params)
        else:
            params = torch.cat((temporal_params, hierarchical_params, ref_y_l, ref_y_r), dim=1)
            params = self.y_prior_fusion_adaptor_2(params)            
        params = self.y_prior_fusion(params)
        return params

    def get_recon_and_feature(self, y_hat, context1_l, context2_l, context3_l, context1_r, context2_r, context3_r, y_q_dec):
        recon_image_feature = self.contextual_decoder(y_hat, context1_l, context2_l, context3_l, context1_r, context2_r, context3_r, y_q_dec)
        feature, x_hat = self.recon_generation_net(recon_image_feature)
        x_hat = x_hat.clamp_(0, 1)
        return x_hat, feature

    def motion_estimation_and_mv_encoding(self, x,ref_x_l,ref_x_r, dpb_l, dpb_r, mv_y_q_enc):
        est_mv_x_l = self.optic_flow(x, ref_x_l)
        est_mv_x_r = self.optic_flow(x, ref_x_r)
        ref_mv_l_r = self.optic_flow(ref_x_l, ref_x_r)
        ref_mv_r_l = self.optic_flow(ref_x_r, ref_x_l)
        mv_y = self.mv_encoder(est_mv_x_l, est_mv_x_r, ref_mv_l_r, ref_mv_r_l, dpb_l["ref_mv_feature"], dpb_r["ref_mv_feature"],  mv_y_q_enc)
        return mv_y, ref_mv_l_r, ref_mv_r_l

    def get_q_for_inference(self, q_in_ckpt, q_index):
        mv_y_q_scale_enc = self.mv_y_q_scale_enc if q_in_ckpt else self.mv_y_q_scale_enc_fine
        mv_y_q_enc = self.get_curr_q(mv_y_q_scale_enc, self.mv_y_q_basic_enc, q_index=q_index)
        mv_y_q_scale_dec = self.mv_y_q_scale_dec if q_in_ckpt else self.mv_y_q_scale_dec_fine
        mv_y_q_dec = self.get_curr_q(mv_y_q_scale_dec, self.mv_y_q_basic_dec, q_index=q_index)

        y_q_scale_enc = self.y_q_scale_enc if q_in_ckpt else self.y_q_scale_enc_fine
        y_q_enc = self.get_curr_q(y_q_scale_enc, self.y_q_basic_enc, q_index=q_index)
        y_q_scale_dec = self.y_q_scale_dec if q_in_ckpt else self.y_q_scale_dec_fine
        y_q_dec = self.get_curr_q(y_q_scale_dec, self.y_q_basic_dec, q_index=q_index)
        return mv_y_q_enc, mv_y_q_dec, y_q_enc, y_q_dec

    def compress(self, x, dpb_l, dpb_r, q_in_ckpt, q_index, layer_id):
        ref_x_l = dpb_l["ref_frame"]
        ref_x_r = dpb_r["ref_frame"]
        mv_y_q_enc, mv_y_q_dec, y_q_enc, y_q_dec = self.get_q_for_inference(q_in_ckpt, q_index)
        mv_y, ref_mv_l_r, ref_mv_r_l = self.motion_estimation_and_mv_encoding(x,ref_x_l,ref_x_r, dpb_l, dpb_r, mv_y_q_enc)
        mv_y_pad, slice_shape = self.pad_for_y(mv_y)
        mv_z = self.mv_hyper_prior_encoder(mv_y_pad)
        mv_z_hat = torch.round(mv_z)
        mv_params = self.mv_prior_param_decoder(mv_z_hat, ref_mv_l_r, ref_mv_r_l, slice_shape)
        mv_y_q_w_0, mv_y_q_w_1, mv_y_q_w_2, mv_y_q_w_3, \
            mv_scales_w_0, mv_scales_w_1, mv_scales_w_2, mv_scales_w_3, mv_y_hat = \
            self.compress_four_part_prior(
                mv_y, mv_params,
                self.mv_y_spatial_prior_adaptor_1, self.mv_y_spatial_prior_adaptor_2,
                self.mv_y_spatial_prior_adaptor_3, self.mv_y_spatial_prior)

        mv_hat_x_l, mv_hat_x_r, mv_feature = self.mv_decoder(mv_y_hat, ref_mv_l_r, ref_mv_r_l, mv_y_q_dec)
        context1_l, context2_l, context3_l, context1_r, context2_r, context3_r = \
            self.motion_compensation(dpb_l, dpb_r, ref_x_l, ref_x_r, mv_hat_x_l, mv_hat_x_r, \
                                     ref_mv_l_r, ref_mv_r_l, layer_id)
        y = self.contextual_encoder(x, context1_l, context2_l, context3_l, context1_r, context2_r, context3_r, y_q_enc)
        y_pad, slice_shape = self.pad_for_y(y)
        z = self.contextual_hyper_prior_encoder(y_pad)
        z_hat = torch.round(z)
        params = self.res_prior_param_decoder(z_hat, dpb_l, dpb_r, context3_l, context3_r, slice_shape)
        y_q_w_0, y_q_w_1, y_q_w_2, y_q_w_3, \
            scales_w_0, scales_w_1, scales_w_2, scales_w_3, y_hat = \
            self.compress_four_part_prior(
                y, params, self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
                self.y_spatial_prior_adaptor_3, self.y_spatial_prior)

        self.entropy_coder.reset()
        self.bit_estimator_z_mv.encode(mv_z_hat)
        self.bit_estimator_z.encode(z_hat)
        self.gaussian_encoder.encode(mv_y_q_w_0, mv_scales_w_0)
        self.gaussian_encoder.encode(mv_y_q_w_1, mv_scales_w_1)
        self.gaussian_encoder.encode(mv_y_q_w_2, mv_scales_w_2)
        self.gaussian_encoder.encode(mv_y_q_w_3, mv_scales_w_3)
        self.gaussian_encoder.encode(y_q_w_0, scales_w_0)
        self.gaussian_encoder.encode(y_q_w_1, scales_w_1)
        self.gaussian_encoder.encode(y_q_w_2, scales_w_2)
        self.gaussian_encoder.encode(y_q_w_3, scales_w_3)
        self.entropy_coder.flush()

        x_hat, feature = self.get_recon_and_feature(y_hat, context1_l, context2_l, context3_l, context1_r, context2_r, context3_r, y_q_dec)
        bit_stream = self.entropy_coder.get_encoded_stream()

        result = {
            "dpb": {
                "ref_frame": x_hat,
                "ref_feature": feature,
                "ref_mv_feature":mv_feature,
                "ref_y": y_hat,
            },
            "bit_stream": bit_stream,
        }
        return result

    def decompress(self, dpb_l, dpb_r, string, height, width, q_in_ckpt, q_index, layer_id):
        _, mv_y_q_dec, _, y_q_dec = self.get_q_for_inference(q_in_ckpt, q_index)

        self.entropy_coder.set_stream(string)
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        z_size = get_downsampled_shape(height, width, 64)
        y_height, y_width = get_downsampled_shape(height, width, 16)
        slice_shape = self.get_to_y_slice_shape(y_height, y_width)
        mv_z_hat = self.bit_estimator_z_mv.decode_stream(z_size, dtype, device)
        z_hat = self.bit_estimator_z.decode_stream(z_size, dtype, device)
        ref_x_l = dpb_l["ref_frame"]
        ref_x_r = dpb_r["ref_frame"]
        ref_mv_l_r = self.optic_flow(ref_x_l, ref_x_r)
        ref_mv_r_l = self.optic_flow(ref_x_r, ref_x_l)        
        mv_params = self.mv_prior_param_decoder(mv_z_hat, ref_mv_l_r, ref_mv_r_l, slice_shape)
        mv_y_hat = self.decompress_four_part_prior(mv_params,
                                                   self.mv_y_spatial_prior_adaptor_1,
                                                   self.mv_y_spatial_prior_adaptor_2,
                                                   self.mv_y_spatial_prior_adaptor_3,
                                                   self.mv_y_spatial_prior)

        mv_hat_x_l, mv_hat_x_r, mv_feature = self.mv_decoder(mv_y_hat, ref_mv_l_r, ref_mv_r_l, mv_y_q_dec)

        context1_l, context2_l, context3_l, context1_r, context2_r, context3_r = \
            self.motion_compensation(dpb_l, dpb_r, ref_x_l, ref_x_r, mv_hat_x_l, mv_hat_x_r, \
                                     ref_mv_l_r, ref_mv_r_l, layer_id)
        params = self.res_prior_param_decoder(z_hat, dpb_l, dpb_r, context3_l, context3_r, slice_shape)
        y_hat = self.decompress_four_part_prior(params,
                                                self.y_spatial_prior_adaptor_1,
                                                self.y_spatial_prior_adaptor_2,
                                                self.y_spatial_prior_adaptor_3,
                                                self.y_spatial_prior)
        x_hat, feature = self.get_recon_and_feature(y_hat, context1_l, context2_l, context3_l, context1_r, context2_r, context3_r, y_q_dec)

        return {
            "dpb": {
                "ref_frame": x_hat,
                "ref_feature": feature,
                "ref_mv_feature":mv_feature,
                "ref_y": y_hat,
            },
        }

    def encode_decode(self, x, dpb_l, dpb_r, layer_id, q_in_ckpt, q_index, output_path=None, pic_width=None, pic_height=None):
        if output_path is not None:
            device = x.device
            torch.cuda.synchronize(device=device)
            t0 = time.time()
            encoded = self.compress(x, dpb_l, dpb_r, q_in_ckpt, q_index, layer_id)
            encode_p(encoded['bit_stream'], q_in_ckpt, q_index, output_path)
            bits = filesize(output_path) * 8
            torch.cuda.synchronize(device=device)
            t1 = time.time()
            q_in_ckpt, q_index, string = decode_p(output_path)

            decoded = self.decompress(dpb_l, dpb_r, string, pic_height, pic_width, q_in_ckpt, q_index, layer_id)
            torch.cuda.synchronize(device=device)
            t2 = time.time()
            result = {
                "dpb": decoded["dpb"],
                "bit": bits,
                "encoding_time": t1 - t0,
                "decoding_time": t2 - t1,
            }
            return result

        encoded = self.forward_one_frame(x, dpb_l, dpb_r, layer_id, q_in_ckpt=q_in_ckpt, q_index=q_index)
        result = {
            "dpb": encoded['dpb'],
            "bit": encoded['bit'].item(),
            "encoding_time": 0,
            "decoding_time": 0,
        }
        return result

    def forward_one_frame(self, x, dpb_l, dpb_r, layer_id, q_in_ckpt=False, q_index=None):
        mv_y_q_enc, mv_y_q_dec, y_q_enc, y_q_dec = self.get_q_for_inference(q_in_ckpt, q_index)
        ref_x_l = dpb_l["ref_frame"]
        ref_x_r = dpb_r["ref_frame"]
        est_mv_x_l = self.optic_flow(x, ref_x_l)
        est_mv_x_r = self.optic_flow(x, ref_x_r)
        ref_mv_l_r = self.optic_flow(ref_x_l, ref_x_r)
        ref_mv_r_l = self.optic_flow(ref_x_r, ref_x_l)

        mv_y = self.mv_encoder(est_mv_x_l, est_mv_x_r, ref_mv_l_r, ref_mv_r_l, dpb_l["ref_mv_feature"], dpb_r["ref_mv_feature"],  mv_y_q_enc)
        mv_y_pad, slice_shape = self.pad_for_y(mv_y)
        mv_z = self.mv_hyper_prior_encoder(mv_y_pad)
        mv_z_hat = self.quant(mv_z)
        mv_params = self.mv_prior_param_decoder(mv_z_hat, ref_mv_l_r, ref_mv_r_l, slice_shape)
        mv_y_res, mv_y_q, mv_y_hat, mv_scales_hat = self.forward_four_part_prior(
            mv_y, mv_params, self.mv_y_spatial_prior_adaptor_1, self.mv_y_spatial_prior_adaptor_2,
            self.mv_y_spatial_prior_adaptor_3, self.mv_y_spatial_prior)
        mv_hat_x_l, mv_hat_x_r, mv_feature = self.mv_decoder(mv_y_hat, ref_mv_l_r, ref_mv_r_l, mv_y_q_dec)
        context1_l, context2_l, context3_l, context1_r, context2_r, context3_r = \
            self.motion_compensation(dpb_l, dpb_r, ref_x_l, ref_x_r, mv_hat_x_l, mv_hat_x_r, \
                                     ref_mv_l_r, ref_mv_r_l, layer_id)
        y = self.contextual_encoder(x, context1_l, context2_l, context3_l, context1_r, context2_r, context3_r, y_q_enc)
        y_pad, slice_shape = self.pad_for_y(y)
        z = self.contextual_hyper_prior_encoder(y_pad)
        z_hat = self.quant(z)
        params = self.res_prior_param_decoder(z_hat, dpb_l, dpb_r, context3_l, context3_r, slice_shape)
        y_res, y_q, y_hat, scales_hat = self.forward_four_part_prior(
            y, params, self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
            self.y_spatial_prior_adaptor_3, self.y_spatial_prior)
        x_hat, feature = self.get_recon_and_feature(y_hat, context1_l, context2_l, context3_l, context1_r, context2_r, context3_r, y_q_dec)

        _, _, H, W = x.size()
        pixel_num = H * W
        y_for_bit = y_q
        mv_y_for_bit = mv_y_q
        z_for_bit = z_hat
        mv_z_for_bit = mv_z_hat
        bits_y = self.get_y_laplace_bits(y_for_bit, scales_hat)
        bits_mv_y = self.get_y_laplace_bits(mv_y_for_bit, mv_scales_hat)
        bits_z = self.get_z_bits(z_for_bit, self.bit_estimator_z)
        bits_mv_z = self.get_z_bits(mv_z_for_bit, self.bit_estimator_z_mv)

        bpp_y = torch.sum(bits_y, dim=(1, 2, 3)) / pixel_num
        bpp_z = torch.sum(bits_z, dim=(1, 2, 3)) / pixel_num
        bpp_mv_y = torch.sum(bits_mv_y, dim=(1, 2, 3)) / pixel_num
        bpp_mv_z = torch.sum(bits_mv_z, dim=(1, 2, 3)) / pixel_num

        bpp = bpp_y + bpp_z + bpp_mv_y + bpp_mv_z
        bit = torch.sum(bpp) * pixel_num
        return {
                "bpp": bpp,
                "dpb": {
                    "ref_frame": x_hat,
                    "ref_feature": feature,
                    "ref_mv_feature":mv_feature,
                    "ref_y": y_hat,
                },
                "bit": bit,
                }
