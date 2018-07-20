
import mxnet as mx
from symbols.symbol import Symbol
import numpy as np


DEEPV_ARM_NET_CONFIG = {
    # f, p, k, s, g
    'dw_cfgs': [
        (16, 1, 3, 2, 16), # conv2
        (32, 1, 3, 1, 32), # conv3
        (32, 1, 3, 1, 32), # conv4
        (32, 1, 3, 2, 32), # conv5
        (64, 1, 3, 1, 64), # conv6
        (64, 1, 3, 1, 64), # conv7
        (64, 1, 3, 1, 64), # conv8
        (64, 1, 3, 1, 64), # conv9
        # (128, 1, 3, 1, 128), # conv10
        # (48, 1, 3, 2, 48), # conv11
        # (48, 1, 3, 2, 48), # conv12
        # (48, 1, 3, 2, 48), # conv13
        # (48, 1, 3, 2, 48), # conv14
    ],

    # f, p, k, s
    'pw_cfgs': [
        (32, 0, 1, 1), # conv2
        (32, 0, 1, 1), # conv3
        (32, 0, 1, 1), # conv4
        (64, 0, 1, 1), # conv5
        (64, 0, 1, 1), # conv6
        (64, 0, 1, 1), # conv7
        (64, 0, 1, 1), # conv8
        (128, 0, 1, 1), # conv9
        # (48, 0, 1, 1), # conv10 -> name: conv10_pw_relu -> feature1, stride 8 
        # (48, 0, 1, 1), # conv11 -> name: conv11_pw_relu -> feature2, stride 16
        # (48, 0, 1, 1), # conv12 -> name: conv12_pw_relu -> feature3, stride 32
        # (48, 0, 1, 1), # conv13 -> name: conv13_pw_relu -> feature3, stride 32
        # (48, 0, 1, 1), # conv14 -> name: conv14_pw_relu -> feature3, stride 32
    ]
}

def checkpoint_callback(bbox_param_names, prefix, means, stds):
    def _callback(iter_no, sym, arg, aux):
        weight = arg[bbox_param_names[0]]
        bias = arg[bbox_param_names[1]]
        stds = np.array([0.1, 0.1, 0.2, 0.2])
        arg[bbox_param_names[0] + '_test'] = (weight.T * mx.nd.array(stds)).T
        arg[bbox_param_names[1] + '_test'] = bias * mx.nd.array(stds)
        mx.model.save_checkpoint(prefix, iter_no + 1, sym, arg, aux)
        arg.pop(bbox_param_names[0] + '_test')
        arg.pop(bbox_param_names[1] + '_test')

    return _callback

class deepv_arm_net(Symbol):
    def __init__(self, n_proposals=400, momentum=0.95, fix_bn=False, test_nbatch=1):
        self.config = DEEPV_ARM_NET_CONFIG
        self.test_nbatch = test_nbatch

    def get_bbox_param_names(self):
        return ['bbox_pred_weight', 'bbox_pred_bias']

    def conv_block(self, data, name, conv_type, num_filter=32, pad=0, kernel=3, stride=1, group=1):
        conv = mx.symbol.Convolution(data=data, num_filter=num_filter, pad=(pad, pad), kernel=(kernel, kernel), 
                                    stride=(stride, stride), num_group=group, name="{}_{}".format(name, conv_type))
        _bn = mx.symbol.BatchNorm(data=conv, use_global_stats=False, name="{}_{}_bn".format(name, conv_type))
        _relu = mx.symbol.Activation(data=_bn, act_type='relu', name="{}_{}_relu".format(name, conv_type))
        return _relu

    def block(self, data, name, dw_config, pw_config):
        f, p, k, s, g = dw_config
        dw = self.conv_block(data, name=name, conv_type="dw", num_filter=f, pad=p, kernel=k, stride=s, group=g)
        f, p, k, s = pw_config
        pw = self.conv_block(dw, name=name, conv_type='pw', num_filter=f, pad=p, kernel=k, stride=s)
        return pw


    def get_deepv_arm_net(self, data):
        # first conv
        conv1 = mx.symbol.Convolution(data, num_filter=16, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv1")
        relu1 = mx.symbol.Activation(data=conv1, act_type='relu', name='relu1')
        n = 1
        for dw_cfg, pw_cfg in zip(self.config['dw_cfgs'], self.config['pw_cfgs']):
            if n == 1:
                block_node = relu1
            n += 1 
            name = 'conv' + str(n)
            block_node = self.block(block_node, name, dw_cfg, pw_cfg)
        return block_node

    def get_rpn(self, conv_feat, num_anchors):
        rpn_conv = mx.sym.Convolution(
            data=conv_feat, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
        rpn_relu = mx.sym.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
        rpn_cls_score = mx.sym.Convolution(
            data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
        rpn_bbox_pred = mx.sym.Convolution(
            data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")
        return rpn_cls_score, rpn_bbox_pred

    def get_symbol_rpn(self, cfg, is_train=True):
        num_anchors = cfg.network.NUM_ANCHORS
        # input init
        if is_train:
            data = mx.sym.Variable(name="data")
            rpn_label = mx.sym.Variable(name='label')
            rpn_bbox_target = mx.sym.Variable(name='bbox_target')
            rpn_bbox_weight = mx.sym.Variable(name='bbox_weight')
            gt_boxes = mx.sym.Variable(name='gt_boxes')
            valid_ranges = mx.sym.Variable(name='valid_ranges')
            im_info = mx.sym.Variable(name='im_info')
        else:
            data = mx.sym.Variable(name="data")
            im_info = mx.sym.Variable(name='im_info')
            im_ids = mx.sym.Variable(name='im_ids')
        # shared convolutional layers
        relu1 = self.get_deepv_arm_net(data) # use conv9

        rpn_cls_score, rpn_bbox_pred = self.get_rpn(relu1, num_anchors)
        rpn_cls_score_reshape = mx.sym.Reshape(data=rpn_cls_score, shape=(0, 2, -1, 0),
                                               name="rpn_cls_score_reshape")
        if is_train:
            # prepare rpn data
            if cfg.TRAIN.fp16 == True:
                grad_scale = float(cfg.TRAIN.scale)
            else:
                grad_scale = 1.0

            # classification
            rpn_cls_prob = mx.sym.SoftmaxOutput(data=rpn_cls_score_reshape, label=rpn_label, multi_output=True,
                                                normalization='valid', use_ignore=True, ignore_label=-1,
                                                name="rpn_cls_prob", grad_scale=grad_scale)

            if cfg.TRAIN.fp16 == True:
                grad_scale = float(cfg.TRAIN.scale)
            else:
                grad_scale = 1.0


            # bounding box regression
            rpn_bbox_loss_ = rpn_bbox_weight * mx.sym.smooth_l1(name='rpn_bbox_loss_', scalar=1.0,
                                                                data=(rpn_bbox_pred - rpn_bbox_target))
            rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_,
                                            grad_scale=3 * grad_scale / float(
                                                cfg.TRAIN.BATCH_IMAGES * cfg.TRAIN.RPN_BATCH_SIZE))
            group = mx.sym.Group([rpn_cls_prob, rpn_bbox_loss])
        else:
            # ROI Proposal
            rpn_cls_prob = mx.sym.SoftmaxActivation(
                data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
            rpn_cls_prob_reshape = mx.sym.Reshape(
                data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')

            rois, rpn_scores = mx.sym.MultiProposal(cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info,
                                        name='rois', batch_size=self.test_nbatch,
                                        rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N,
                                        rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                                        rpn_min_size=cfg.TEST.RPN_MIN_SIZE,
                                        threshold=cfg.TEST.RPN_NMS_THRESH,
                                        feature_stride=cfg.network.RPN_FEAT_STRIDE,
                                        ratios=tuple(cfg.network.ANCHOR_RATIOS),
                                        scales=tuple(cfg.network.ANCHOR_SCALES))

            group = mx.sym.Group([rois, rpn_scores, im_ids])

        self.sym = group
        return group

    def get_symbol_rpn_ugly(self, cfg, is_train=True):
        num_anchors = cfg.network.NUM_ANCHORS

        # input init
        if is_train:
            data = mx.sym.Variable(name="data")
            rpn_label = mx.sym.Variable(name='label')
            rpn_bbox_target = mx.sym.Variable(name='bbox_target')
            rpn_bbox_weight = mx.sym.Variable(name='bbox_weight')
            gt_boxes = mx.sym.Variable(name='gt_boxes')
            valid_ranges = mx.sym.Variable(name='valid_ranges')
            im_info = mx.sym.Variable(name='im_info')
        else:
            data = mx.sym.Variable(name="data")
            im_info = mx.sym.Variable(name='im_info')
            im_ids = mx.sym.Variable(name='im_ids')

        relu1 = self.get_deepv_arm_net(data) # use conv9

        rpn_cls_score, rpn_bbox_pred = self.get_rpn(relu1, num_anchors)

        conv_new_1 = mx.sym.Convolution(data=relu1, kernel=(1, 1), num_filter=256, name="conv_new_1")
        conv_new_1_relu = mx.sym.Activation(data=conv_new_1, act_type='relu', name='conv_new_1_relu')

        rpn_cls_score_reshape = mx.sym.Reshape(data=rpn_cls_score, shape=(0, 2, -1, 0),
                                               name="rpn_cls_score_reshape")
        if is_train:
            # prepare rpn data
            if cfg.TRAIN.fp16 == True:
                grad_scale = float(cfg.TRAIN.scale)
            else:
                grad_scale = 1.0

            # classification
            rpn_cls_prob = mx.sym.SoftmaxOutput(data=rpn_cls_score_reshape, label=rpn_label, multi_output=True,
                                                normalization='valid', use_ignore=True, ignore_label=-1,
                                                name="rpn_cls_prob", grad_scale=grad_scale)

            rois, label, bbox_target, bbox_weight = mx.sym.MultiProposalTarget(cls_prob=rpn_cls_prob, bbox_pred=rpn_bbox_pred, im_info=im_info,
                                                                               gt_boxes=gt_boxes, valid_ranges=valid_ranges, batch_size=cfg.TRAIN.BATCH_IMAGES, name='multi_proposal_target')
            label = mx.symbol.Reshape(data=label, shape=(-1,), name='label_reshape')
            offset_t = mx.contrib.sym.DeformablePSROIPooling(name='offset_t', data=conv_new_1_relu, rois=rois, group_size=1, pooled_size=7,
                                                             sample_per_part=4, no_trans=True, part_size=7, output_dim=256, spatial_scale=0.0625)
            offset = mx.sym.FullyConnected(name='offset', data=offset_t, num_hidden=7 * 7 * 2, lr_mult=0.01)
            offset_reshape = mx.sym.Reshape(data=offset, shape=(-1, 2, 7, 7), name="offset_reshape")

            deformable_roi_pool = mx.contrib.sym.DeformablePSROIPooling(name='deformable_roi_pool', data=conv_new_1_relu, rois=rois,
                                                                        trans=offset_reshape, group_size=1, pooled_size=7, sample_per_part=4,
                                                                        no_trans=False, part_size=7, output_dim=256, spatial_scale=0.0625, trans_std=0.1)
            # 2 fc
            fc_new_1 = mx.sym.FullyConnected(name='fc_new_1', data=deformable_roi_pool, num_hidden=1024)
            fc_new_1_relu = mx.sym.Activation(data=fc_new_1, act_type='relu', name='fc_new_1_relu')

            fc_new_2 = mx.sym.FullyConnected(name='fc_new_2', data=fc_new_1_relu, num_hidden=1024)
            fc_new_2_relu = mx.sym.Activation(data=fc_new_2, act_type='relu', name='fc_new_2_relu')
            num_classes = cfg.dataset.NUM_CLASSES
            num_reg_classes = 1
            cls_score = mx.sym.FullyConnected(name='cls_score', data=fc_new_2_relu, num_hidden=num_classes)
            bbox_pred = mx.sym.FullyConnected(name='bbox_pred', data=fc_new_2_relu, num_hidden=num_reg_classes * 4)

            if cfg.TRAIN.fp16 == True:
                grad_scale = float(cfg.TRAIN.scale)
            else:
                grad_scale = 1.0

            cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='valid', use_ignore=True, ignore_label=-1,
                                            grad_scale=grad_scale)
            bbox_loss_ = bbox_weight * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0,
                                                        data=(bbox_pred - bbox_target))
            bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=grad_scale / (188.0*16.0))
            rcnn_label = label

            # reshape output
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TRAIN.BATCH_IMAGES, -1, num_classes),
                                      name='cls_prob_reshape')
            bbox_loss = mx.sym.Reshape(data=bbox_loss, shape=(cfg.TRAIN.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                       name='bbox_loss_reshape')


            # bounding box regression
            rpn_bbox_loss_ = rpn_bbox_weight * mx.sym.smooth_l1(name='rpn_bbox_loss_', scalar=1.0,
                                                                data=(rpn_bbox_pred - rpn_bbox_target))
            rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_,
                                            grad_scale=3 * grad_scale / float(
                                                cfg.TRAIN.BATCH_IMAGES * cfg.TRAIN.RPN_BATCH_SIZE))

            #group = mx.sym.Group([rpn_cls_prob, rpn_bbox_loss, mx.sym.BlockGrad(cls_prob), mx.sym.BlockGrad(bbox_loss), mx.sym.BlockGrad(rcnn_label)])
            group = mx.sym.Group([rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, mx.sym.BlockGrad(rcnn_label)])

        else:
            # ROI Proposal
            rpn_cls_prob = mx.sym.SoftmaxActivation(
                data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
            rpn_cls_prob_reshape = mx.sym.Reshape(
                data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')

            rois, rpn_scores = mx.sym.MultiProposal(cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info,
                                        name='rois', batch_size=self.test_nbatch,
                                        rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N,
                                        rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                                        rpn_min_size=cfg.TEST.RPN_MIN_SIZE,
                                        threshold=cfg.TEST.RPN_NMS_THRESH,
                                        feature_stride=cfg.network.RPN_FEAT_STRIDE,
                                        ratios=tuple(cfg.network.ANCHOR_RATIOS),
                                        scales=tuple(cfg.network.ANCHOR_SCALES))

            offset_t = mx.contrib.sym.DeformablePSROIPooling(name='offset_t', data=conv_new_1_relu, rois=rois,
                                                             group_size=1, pooled_size=7,
                                                             sample_per_part=4, no_trans=True, part_size=7,
                                                             output_dim=256, spatial_scale=0.0625)
            offset = mx.sym.FullyConnected(name='offset', data=offset_t, num_hidden=7 * 7 * 2, lr_mult=0.01)
            offset_reshape = mx.sym.Reshape(data=offset, shape=(-1, 2, 7, 7), name="offset_reshape")

            deformable_roi_pool = mx.contrib.sym.DeformablePSROIPooling(name='deformable_roi_pool',
                                                                        data=conv_new_1_relu, rois=rois,
                                                                        trans=offset_reshape, group_size=1,
                                                                        pooled_size=7, sample_per_part=4,
                                                                        no_trans=False, part_size=7, output_dim=256,
                                                                        spatial_scale=0.0625, trans_std=0.1)
            # 2 fc
            fc_new_1 = mx.sym.FullyConnected(name='fc_new_1', data=deformable_roi_pool, num_hidden=1024)
            fc_new_1_relu = mx.sym.Activation(data=fc_new_1, act_type='relu', name='fc_new_1_relu')

            fc_new_2 = mx.sym.FullyConnected(name='fc_new_2', data=fc_new_1_relu, num_hidden=1024)
            fc_new_2_relu = mx.sym.Activation(data=fc_new_2, act_type='relu', name='fc_new_2_relu')
            num_classes = cfg.dataset.NUM_CLASSES
            num_reg_classes = 1
            cls_score = mx.sym.FullyConnected(name='cls_score', data=fc_new_2_relu, num_hidden=num_classes)
            bbox_pred = mx.sym.FullyConnected(name='bbox_pred', data=fc_new_2_relu, num_hidden=num_reg_classes * 4)
            cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(self.test_nbatch, -1, num_classes),
                                      name='cls_prob_reshape')
            bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(self.test_nbatch, -1, 4 * num_reg_classes),
                                       name='bbox_pred_reshape')

#            group = mx.sym.Group([rois, cls_prob, im_ids])
            group = mx.sym.Group([rpn_scores, rois, cls_prob, bbox_pred, im_ids])

        self.sym = group
        return group

    def init_weight_rcnn(self, cfg, arg_params, aux_params):
        pass

    def init_weight_rpn(self, cfg, arg_params, aux_params):
        pass

    def init_weight(self, cfg, arg_params, aux_params):
        self.init_weight_rcnn(cfg, arg_params, aux_params)
