class CSP_S(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5): # 1024 512 3 False 1 0.5
        super().__init__()
        c_ = int(c2 * e) # 256
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        #self.cv2 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck_S(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        x = torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1)
        b, c, h, w = x.shape
        groups = 8
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b, -1, h, w)
        return x

class Bottleneck_S(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DWConv(c_, c2, 3, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.add = shortcut and c1 == c2
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))