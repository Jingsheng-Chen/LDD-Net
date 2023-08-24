class CPSV2(nn.Module):
    def __init__(self, c1, c2, e=0.5, n=1, g=1):  # ch_in, ch_out, expansion, number , groups,
        super().__init__()
        c_ = int(c1 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1) #-6
        self.cv2 = Conv(c1, c_, 1, 1) #-5

        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 3, 1) #-3
        self.cv5 = Conv(c_, c_, 3, 1)
        self.cv6 = Conv(c_, c_, 3, 1) #-1

        self.cv7 = Conv(4 * c_, c2, 1, 1)


    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)

        x3 = self.cv4(self.cv3(x2))
        x4 = self.cv6(self.cv5(x3))

        x = torch.cat((x1, x2, x3, x4), 1)

        return self.cv7(x)