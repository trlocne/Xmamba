import torch 
from xmamba.xmamba import XMamba

t1 = torch.rand(1, 4, 128, 128, 128).cuda()


model = XMamba(in_chans=4,
                 out_chans=4,
                 depths=[2,2,2,2],
                 feat_size=[64, 128, 256, 512]).cuda()

out = model(t1)

print(out.shape)