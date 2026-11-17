"""MorphSymNet: a compact segmentation network on regulated-morphology and symmetry features.

    image -> FeatureFrontEnd (fixed, no parameters)        53 planes, full resolution
          -> stem: two stride-2 3x3 convs                  1/4 (the first one mixes the planes)
             (inject='quarter': RGB stem + a side stem for the other planes, added at 1/4)
          -> encoder: residual stages at 1/4, 1/8, 1/16    64 / 128 / 256 channels
          -> pyramid pooling context module                1/16
          -> decoder: two top-down fusions with skips      1/8, 1/4
          -> classifier, bilinear x4                       full resolution

    learn_diff: RGB -> LearnableDiffFilter (generalFilter with trained parameters), added to
                the stem at 1/2 (or 1/4) through a zero-initialised 1x1 projection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import FeatureFrontEnd
from .guided_upsample import GuidedUpsample
from .learnable_diff import LearnableDiffFilter


def conv_bn_relu(c_in, c_out, k=3, stride=1, dilation=1):
    pad = dilation * (k // 2)
    return nn.Sequential(nn.Conv2d(c_in, c_out, k, stride, pad, dilation, bias=False),
                         nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class BasicBlock(nn.Module):
    """Two 3x3 convolutions with a residual connection (ResNet basic block)."""

    def __init__(self, c_in, c_out, stride=1, dilation=1):
        super().__init__()
        self.conv1 = conv_bn_relu(c_in, c_out, 3, stride, dilation)
        self.conv2 = nn.Sequential(nn.Conv2d(c_out, c_out, 3, 1, dilation, dilation, bias=False),
                                   nn.BatchNorm2d(c_out))
        self.shortcut = None
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False),
                                          nn.BatchNorm2d(c_out))

    def forward(self, x):
        identity = x if self.shortcut is None else self.shortcut(x)
        return F.relu(self.conv2(self.conv1(x)) + identity, inplace=True)


class PyramidPooling(nn.Module):
    """Global context: the feature map is average-pooled to a few coarse grids, each is
    projected, upsampled and added back (the lightweight pyramid pooling of PP-LiteSeg)."""

    def __init__(self, c_in, c_out, bins=(1, 2, 4)):
        super().__init__()
        self.stages = nn.ModuleList(nn.Sequential(nn.AdaptiveAvgPool2d(b), nn.Conv2d(c_in, c_out, 1, bias=False),
                                                  nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)) for b in bins)
        self.local = conv_bn_relu(c_in, c_out, 1)
        self.out = conv_bn_relu(c_out, c_out, 3)

    def forward(self, x):
        y = self.local(x)
        for stage in self.stages:
            y = y + F.interpolate(stage(x), size=x.shape[2:], mode='bilinear', align_corners=False)
        return self.out(y)


class Fuse(nn.Module):
    """Top-down decoder step: upsample the coarse map, add the projected skip, smooth."""

    def __init__(self, c_coarse, c_skip, c_out):
        super().__init__()
        self.coarse = conv_bn_relu(c_coarse, c_out, 1)
        self.skip = conv_bn_relu(c_skip, c_out, 1)
        self.out = conv_bn_relu(c_out, c_out, 3)

    def forward(self, coarse, skip):
        up = F.interpolate(self.coarse(coarse), size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.out(up + self.skip(skip))


class MorphSymNet(nn.Module):
    """Segmentation network. forward() takes an RGB batch (B, 3, H, W) in [0, 1] and returns
    logits (B, n_classes, H, W).

    features: the input groups, any of rgb, morph, sym, haar, diff.
    inject:   'input'   - all planes go into the first convolution;
              'quarter' - only RGB goes into the stem; the other planes pass through a side
                          stem of their own and are ADDED to the stem output at 1/4
                          resolution. The side stem ends in a zero-initialised BatchNorm, so
                          training starts as the plain RGB network.
    gate:     every non-RGB plane is multiplied by a learnable scalar that starts at 0, so
              training starts as the plain RGB network and the planes enter only as far
              as the loss asks for them. self.gate holds the scalars after training.
    learn_diff: None, or keyword arguments of LearnableDiffFilter. Its planes are computed
              from RGB at 1/2 (stride 2) or 1/4 (stride 4) resolution and ADDED to the stem
              there, through a 1x1 convolution and a zero-initialised BatchNorm: training
              starts as the plain RGB network. 'freeze': True keeps the filters at their
              initial setting (a fixed soft bank), as the control for the learning.
    guided_up: None, or keyword arguments of GuidedUpsample, which then replaces the bilinear
              upsampling of the logits.
    """

    def __init__(self, n_classes=19, features=('rgb', 'morph', 'sym'), widths=(32, 64, 128, 256),
                 backend='cuda', input_norm=False, inject='input', morph_planes='all',
                 morph_kwargs=None, sym_kwargs=None, haar_kwargs=None, diff_bank='full', gate=False,
                 learn_diff=None, guided_up=None):
        super().__init__()
        self.frontend = FeatureFrontEnd('rgb' in features, 'morph' in features, 'sym' in features,
                                        'haar' in features, backend, morph_planes,
                                        morph_kwargs, sym_kwargs, haar_kwargs,
                                        'diff' in features, diff_bank)
        n = self.frontend.n_features
        w1, w2, w3, w4 = widths
        self.gate = None
        if gate:
            if 'rgb' not in features or n <= 3:
                raise ValueError("gate needs rgb and at least one other group")
            self.gate = nn.Parameter(torch.zeros(n - 3))
        self.n_main = n
        self.side_stem = None
        if inject == 'quarter':
            if 'rgb' not in features or n <= 3:
                raise ValueError("inject='quarter' needs rgb and at least one other group")
            self.n_main = 3
            last_bn = nn.BatchNorm2d(w2)
            nn.init.zeros_(last_bn.weight)
            self.side_stem = nn.Sequential(conv_bn_relu(n - 3, w1, 3, 2),
                                           nn.Conv2d(w1, w2, 3, 2, 1, bias=False), last_bn)
        elif inject != 'input':
            raise ValueError("inject must be 'input' or 'quarter'")
        # The first convolution reads all planes at once and halves the resolution, so no
        # full-resolution activation is ever stored. Linear combinations of the quantile
        # planes are top-hats and morphological gradients; the BatchNorm after it absorbs
        # the different ranges of the planes.
        self.stem = nn.Sequential(conv_bn_relu(self.n_main, w1, 3, 2), conv_bn_relu(w1, w2, 3, 2))  # 1/4
        # optional per-plane normalisation of the feature stack (costs a full-resolution pass)
        self.input_norm = nn.BatchNorm2d(n) if input_norm else nn.Identity()
        self.stage1 = nn.Sequential(BasicBlock(w2, w2), BasicBlock(w2, w2))                      # 1/4
        self.stage2 = nn.Sequential(BasicBlock(w2, w3, 2), BasicBlock(w3, w3))                   # 1/8
        self.stage3 = nn.Sequential(BasicBlock(w3, w4, 2), BasicBlock(w4, w4),
                                    BasicBlock(w4, w4, dilation=2), BasicBlock(w4, w4, dilation=4))  # 1/16
        self.context = PyramidPooling(w4, w3)
        self.fuse2 = Fuse(w3, w3, 96)
        self.fuse1 = Fuse(96, w2, 64)
        self.head = nn.Sequential(conv_bn_relu(64, 64, 3), nn.Conv2d(64, n_classes, 1))
        # created last, so that the rest of the network is initialised exactly as without them
        self.gup = GuidedUpsample(**guided_up) if guided_up is not None else None
        if self.gup is not None and 'rgb' not in features:
            raise ValueError("guided_up needs rgb")
        self.ldiff = None
        if learn_diff is not None:
            if 'rgb' not in features:
                raise ValueError("learn_diff needs rgb")
            kw = dict(learn_diff)
            freeze = kw.pop('freeze', False)
            self.ldiff = LearnableDiffFilter(**kw)
            if self.ldiff.stride not in (2, 4):
                raise ValueError("learn_diff: stride must be 2 or 4")
            self.ldiff.requires_grad_(not freeze)
            last_bn = nn.BatchNorm2d(w1 if self.ldiff.stride == 2 else w2)
            nn.init.zeros_(last_bn.weight)
            self.ldiff_proj = nn.Sequential(nn.Conv2d(self.ldiff.n_planes, last_bn.num_features, 1, bias=False),
                                            last_bn)

    def forward_features(self, feats):
        """Network body on a precomputed feature stack."""
        x = self.input_norm(feats)
        if self.gate is not None:
            x = torch.cat([x[:, :3], x[:, 3:] * self.gate.view(1, -1, 1, 1)], 1)
        main = x if self.side_stem is None else x[:, :3]
        if self.ldiff is None:
            s1 = self.stem(main)
        else:
            planes = self.ldiff_proj(self.ldiff(feats[:, :3]))
            if self.ldiff.stride == 2:
                s1 = self.stem[1](self.stem[0](main) + planes)
            else:
                s1 = self.stem(main) + planes
        if self.side_stem is not None:
            s1 = s1 + self.side_stem(x[:, 3:])
        s1 = self.stage1(s1)
        s2 = self.stage2(s1)
        s3 = self.context(self.stage3(s2))
        y = self.head(self.fuse1(self.fuse2(s3, s2), s1))
        if self.gup is not None:
            return self.gup(y, feats[:, :3])
        return F.interpolate(y, size=feats.shape[2:], mode='bilinear', align_corners=False)

    def forward(self, rgb):
        return self.forward_features(self.frontend(rgb))


def build_model(n_classes, features, args):
    """MorphSymNet from the argument dictionary that train.py stores in log.json."""
    hw = args.get('sym_half_widths')
    learn_diff = None
    if args.get('learn_diff'):
        learn_diff = {'stride': args.get('ld_stride', 2), 'spacings': tuple(args.get('ld_spacings') or (2.0,)),
                      'directions': args.get('ld_directions', 8), 'freeze': args.get('ld_freeze', False)}
    return MorphSymNet(n_classes, tuple(features), input_norm=args.get('input_norm', False),
                       inject=args.get('inject', 'input'), morph_planes=args.get('morph_planes', 'all'),
                       sym_kwargs={'half_widths': tuple(hw)} if hw else None,
                       diff_bank=args.get('diff_bank', 'full'), gate=args.get('gate', False),
                       learn_diff=learn_diff, guided_up={} if args.get('guided_up') else None)
