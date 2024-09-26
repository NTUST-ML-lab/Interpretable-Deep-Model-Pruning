import torch


class renyi_estim():
    def __init__(self, layers, n_iterations, DEVICE = "cuda:0") -> None:
        self.DEVICE = DEVICE
        self.sigmas = torch.zeros((layers +2, n_iterations)).to(DEVICE)
        self.softmax = torch.nn.Softmax(dim=1)

    def getLastSigma(self, layerIdx):
        ret_sig = 0
        for si in self.sigmas[layerIdx+1]:
            if si != 0:
                ret_sig =  si
        return ret_sig

    def pairwise_distances(self, x):
        # x 的形狀應該是 (n, d)，其中 n 是點的數量，d 是每個點的維度
        n = x.size(0)
        dist = torch.zeros(n, n, device=x.device)
        
        for i in range(n):
            for j in range(i + 1, n):
                dist[i, j] = torch.norm(x[i] - x[j])
                dist[j, i] = dist[i, j]  # 距離矩陣是對稱的
        
        return dist

    def compute_tensor_size(self, n, d, element_size=4):
        # 計算中間張量的大小，單位是字節
        tensor_size = n * n * d * element_size
        # 轉換成 MB（1 MB = 1024 * 1024 字節）
        tensor_size_mb = tensor_size / (1024 ** 2)
        return tensor_size_mb


    def dist_mat(self, x):

        try:
            x = torch.from_numpy(x)
        except TypeError:
            x = x

        if len(x.size()) == 4:
            x = x.view(x.size()[0], -1)

        if self.compute_tensor_size(x.shape[0], x.shape[1]) > 14000: #torch.cuda.mem_get_info()[0]/(1024**2) :
            dist = self.pairwise_distances(x)
        else:
            dist = torch.norm(x[:, None] - x, dim=2)
        return dist

    
    def kernel_mat(self, x, k_x, k_y, sigma=None, epoch=None, idx=None, factor = 1):

        d = self.dist_mat(x)
        if sigma is None:
            if epoch > 40:
                sigma_vals = torch.linspace(0.1, 10*d.mean().item(), 50).to(self.DEVICE)
            else:
                sigma_vals = torch.linspace(0.1, 10*d.mean().item(), 75).to(self.DEVICE)
            L = []
            for sig in sigma_vals:
                k_l = torch.exp(-d ** 2 / (sig ** 2)) / d.size(0)
                L.append(self.kernel_loss(k_x, k_y, k_l, idx))

            if epoch == 0:
                self.sigmas[idx+1, epoch] = sigma_vals[L.index(max(L))]
            else:
                self.sigmas[idx+1, epoch] = 0.9*self.sigmas[idx+1, epoch-1] + 0.1*sigma_vals[L.index(max(L))]

            sigma = self.sigmas[idx+1, epoch]
        sigma = sigma.to(self.DEVICE)
        return torch.exp(-d ** 2 / (factor * sigma ** 2))


    def kernel_loss(self, k_x, k_y, k_l, idx, tmp = ""):

        b = 1.0
        beta = [b, b, b, b, b]

        L = torch.norm(k_l)
        Y = torch.norm(k_y) ** beta[idx]
        X = torch.norm(k_x) ** (1-beta[idx])

        LY = torch.trace(torch.matmul(k_l, k_y))**beta[idx]
        LX = torch.trace(torch.matmul(k_l, k_x))**(1-beta[idx])

        return 2*torch.log2((LY*LX)/(L*Y*X))

    @staticmethod
    def entropy(*args):

        for idx, val in enumerate(args):
            if idx == 0:
                k = val.clone()
            else:
                k *= val

        k /= k.trace()

        eigv = torch.linalg.eigh(k)[0].abs()

        return -(eigv*(eigv.log2())).sum()
    
    def one_hot(self, y, gpu):

        try:
            y = torch.from_numpy(y)
        except TypeError:
            None

        y_1d = y
        if gpu:
            y_hot = torch.zeros((y.size(0), torch.max(y).int()+1)).to(self.DEVICE)
        else:
            y_hot = torch.zeros((y.size(0), torch.max(y).int()+1))

        for i in range(y.size(0)):
            y_hot[i, y_1d[i].int()] = 1

        return y_hot