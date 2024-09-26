import torch
from pytorch_util import *

# references: 
# [1] Information Plane Analysis of Deep Neural Networks via Matrix-Based Renyi's Entropy and Tensor Kernels
# [2] HRel: Filter pruning based on High Relevance between activation maps and class labels
# [3] Multivariate Extension of Matrix-based Renyi’s ´ α-order Entropy Functional

class sigmaTracer:
    # Choosing the kernel width
    # from paper: [1]

    def __init__(self, layerNames:list): # ["Z1", "Z2"] ... 
        # record the sigma of epoch 0 ~ 1999 which can be increase
        # layer name : sigma of each epoch
        self.layerSigma = {
            layer : torch.zeros(2000) for layer in layerNames
        }

        # index : layer name
        self.layerIndex = {
            layerNames[i] : i for i in range(len(layerNames))
        }
        self.layerNames = layerNames

        # record where is the last epoch
        self.last_iter = 0


    def update(self, value, layer, iteration):
        '''
        Update the sigma 
        value: the sigma calculated in the "iteration"-th epoch for "layer"
        '''

        # transfer the index to layer name
        if type(layer) != str:
            layer = self.layerNames[layer]

        # update the sigma
        if iteration == 0:
            self.layerSigma[layer][iteration] = value
        else:
            self.layerSigma[layer][iteration] = 0.9 * self.layerSigma[layer][iteration-1] + 0.1 * value
            
        # update where is the last epoch
        self.last_iter = self.last_iter if self.last_iter > iteration else iteration

    def get(self, layer, iteration):
        '''
        Get sigma in "iteration"-th epoch of "layer"
        '''
        return self.layerSigma[layer][iteration]
    
    def getLast(self, layer):
        '''
        Get sigma in last epoch of "layer"
        '''
        return self.layerSigma[layer][self.last_iter]
    

    def add(self, layerName):
        '''
        add new layer
        '''
        self.layerSigma[layerName] = torch.zeros(2000)
        self.layerIndex[len(list(self.layerIndex.keys()))] = layerName
    
def dist_mat(x, presion = None):
    '''
    Calculate the exponent of Gram matrix of given x
    Modified from official code [2]
    '''

    # 確保所有操作都是可以使用 GPU
    try:
        x = torch.from_numpy(x)
    except TypeError:
        x = x
        
    # transfer the x to given presion
    if presion:
        x = x.type(presion)

    # 預算好所有xi - xj 的組合
    s = x.shape
    # if len(x.size()) == 4:
    if len(x.size()) > 2:
        x = x.view(x.size()[0], -1)
    dist = x[:, None] - x

    # fab norm
    return torch.norm(dist, dim=2, dtype=presion)


def kernel_loss(k_y, k_l):
    '''
    Return the kernel alignment loss with k_y and k_l [1]
    This is official code [2]
    '''
    beta = 1.0

    L = torch.norm(k_l)
    Y = torch.norm(k_y) ** beta
    #X = th.norm(k_x) ** (1-beta) 

    LY = torch.trace(torch.matmul(k_l, k_y))**beta
    #LX = th.trace(th.matmul(k_l, k_x))**(1-beta)

    #return 2*th.log2((LY*LX)/(L*Y*X))
    return 2*torch.log2((LY)/(L*Y))

def gaussKernel(x, k_y=None , sigmatracer : sigmaTracer = None, sigma = None, iteration = None, layer = None, device = "cuda:0", activate = False, presion = None):
    '''
    Calculate the Gram matrix of x ( e^(−1/σ^2 ∥xi−xj∥_F^2) )
    x: given data, k_y: the target kernel you want to allign
    sigmaTracer: choosing the kernel width in given "iteration" for "layer"
    sigma: calculate the Gram matrix with this sigma
    device: cpu or cuda
    activate: if the given x is one-hot label data or not
    presion: floating presion

    Modified from official code [2]
    '''
    
    # If the given x is one-hot label data, it shall be calculated with Softmax
    if activate:
        softmax = torch.nn.Softmax(dim = 1)
        x = softmax(x)

    # calculate ∥xi−xj∥_F^2 
    d = dist_mat(x).to(device)
    if sigma is None:  # 用類似暴力法的方式找出一個與 ky 最接近的sigma
        if presion:
            k_y = k_y.type(presion)
            
        if iteration > 20:
            sigma_vals = torch.linspace(0.3, 10*d.mean(), 100, dtype=presion).to(device) 
        else:
            sigma_vals = torch.linspace(0.3, 10*d.mean(), 300, dtype=presion).to(device)
        L = []

        # chosing the sigma
        for sig in sigma_vals:
            # x ( e^(−1/σ^2 ∥xi−xj∥_F^2) )
            k_l = torch.exp(-d ** 2 / ( sig ** 2)) / d.size(0) 
            if presion:
                k_l = k_l.type(presion)

            L.append(kernel_loss(k_y, k_l))

        # update the sigma chosen in "iteration"-th epoch
        sigmatracer.update(sigma_vals[L.index(max(L))], layer, iteration)
        sigma = sigmatracer.get(layer, iteration)


    return torch.exp(-d ** 2 / (2*sigma ** 2)) # [3]

def polynomialKernel(x, r=1, p=2):
    '''
    return the polynomial Kernel
    *** Note: this method is deprecated ***
    '''
    s = x.size(0)
    K = (r + torch.matmul(x, x.T)).pow(p)
    return K
    N = torch.zeros( (s, s) )
    for i in range(s):
        for j in range(s):
            N[i][j]= computeNij(K[i][i], K[i][j], K[j][j], s)
    return N

def computeNij(Gii, Gij, Gjj, s):
    '''
    *** Note: this method is deprecated ***
    '''
    return Gij.div(torch.sqrt(Gii * Gjj)).div(s)

def N_matrix(x, k_y=None, sigmatracer : sigmaTracer = None, sigma = None, iteration = None, layer = None, device = "cuda:0", activate = False, presion = None):
    '''
    return the Gram matrix of x
    '''
    G = gaussKernel(x,k_y, sigmatracer, sigma, iteration, layer, device=device, activate=activate, presion = presion)
    return G 

    # *** Note: below codes are deprecated ***
    s = G.size(0)
    N = torch.zeros( (s, s) )
    for i in range(s):
        for j in range(s):
            N[i][j]= computeNij(G[i][i], G[i][j], G[j][j], s)
    return N



def Entropy(N, device = "cuda:0", presion = None):
    '''
    From official code [2]
    *** Note: this method is deprecated since they are not alpha-Renyi entropy***
    '''
    if presion:
        N.type(dtype=presion)
    eigenvalues = torch.abs(torch.symeig(N, eigenvectors=False)[0]).to(device)
    temp=eigenvalues.clone()
    eigv_log2= temp.log2().to(device)
    c = torch.tensor([0]).to(device)
    if((eigenvalues==c).any()):
        zero_indices=(eigenvalues == 0).nonzero().tolist()
        #small=th.tensor([0.999999999]).to(device)
        small=torch.tensor([0.0000000099]).to(device)
        small_value=small.detach().clone()
        for i in zero_indices:
            eigv_log2[i]=small_value
    # a = torch.sum(eigenvalues.matmul(torch.diag(torch.log2(eigenvalues))))
    # b = torch.sum(eigenvalues.mul(torch.log2(eigenvalues) ) ) 
    return -(eigenvalues*(eigv_log2)).sum()

def Entropy2(N, alpha=1.01, device = "cuda:0", presion = None):
    '''
    return the alpha-Renyi entropy of given N [3]
    N: the normalized Gram matrix
    '''
    eigenvalues = (torch.abs(torch.symeig(N, eigenvectors=False)[0])**alpha).to(device, dtype=presion)
    return eigenvalues.sum().log2() / (1-alpha)





def jointEntropy(x, y, sigmaX, sigmaY = None):
    '''
    A simple way to calculate joint entropy of x, y
    *** Note: this method is deprecated ***
    '''
    if sigmaY == None:
        sigmaX = sigmaY
    Nx = N_matrix(x, sigmaX)
    Ny = N_matrix(y, sigmaY)
    hm = Nx.mul(Ny)
    return Entropy(hm / hm.trace())

def MI(X, Y, sigmaX, sigmaY):
    '''
    A simple way to calculate I(X;Y)
    *** Note: this method is deprecated ***
    '''
    HX = Entropy(N_matrix(X, sigmaX))
    HY = Entropy(N_matrix(Y, sigmaY))
    HXY = jointEntropy(X, Y, sigmaX, sigmaY)
    return HX + HY - HXY

if __name__ == "__main__":
    '''
    test code
    '''
    ones = torch.ones((3,3,32,32))
    test = torch.FloatTensor([ [0,1,2,3],[1,2,3,4],[3,4,5,6] ])
    pk = polynomialKernel(test)
    Epk = Entropy2(pk, alpha=2)

    G = gaussKernel(test, sigma = 0.5)
    N = N_matrix(test, sigma = 0.5)
    E = Entropy(N)
    JE = jointEntropy(ones, test, sigmaX = 0.5, sigmaY = 0.5)
    mi = MI(ones, test, 0.5, 0.5)
    pass