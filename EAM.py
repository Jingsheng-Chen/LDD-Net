class EAM(nn.Module):
    def __init__(self, c1 ,c2):
        super(EAM, self).__init__()
        self.CAM = CAM()
        self.SAM = SAM()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        C = self.CAM(x)
        S = self.SAM(x)
        W = torch.add(C,S)
        W = self.sigmoid(W)

        return x * W
