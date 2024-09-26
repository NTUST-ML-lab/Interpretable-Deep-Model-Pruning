import torch
# from torch_cluster import knn
import numpy as np
import scipy.spatial as ss
from scipy.special import digamma
import time
import datetime

BATCHSIZE = 10

def time_monitor(func):
    def wrap(*args, **kwargs):
        print(f"Calculating the function {func.__name__}")
        before = time.time()
        x = func(*args,**kwargs)
        after = time.time()
        TIME = after - before
        print(f"\tCost time: {datetime.timedelta(seconds = (TIME))}")
        x = [x, TIME]
        return x
    return wrap

def ram_monitor(func):
    def wrap(*args, **kwargs):
        device_id = 0
        torch.cuda.reset_max_memory_allocated(0)
        before = torch.cuda.max_memory_allocated(0)
        x = func(*args,**kwargs)
        after = torch.cuda.max_memory_allocated(0)
        USAGE = after - before
        print(f"\tCuda usage: {USAGE/(1024*1024):.2f} MB")
        x.append(USAGE)
        return tuple(x)
    return wrap

############### Base ###############
def unique(xd, dim = None):
    unique_array, unique_inverse, counts = torch.unique(xd,
                                                        return_inverse = True, 
                                                        return_counts = True, dim = dim)
    return unique_array, unique_inverse, counts

def extract_p(xd):
    unique_array, unique_inverse, unique_counts = unique(xd, dim = 0)
    p = unique_counts/ torch.sum(unique_counts, dtype = torch.float)
    
    return p, unique_inverse

def entropy_p(p, base=2):
    return -torch.sum(p * torch.log(p)) / torch.log(torch.tensor(base))

############### kNN ###############
# 本區塊引用自 https://github.com/ravidziv/IDNNs/blob/master/idnns/information/entropy_estimators.py
def entropy(x, k=3, base=2, p=float('inf')):
    """The classic K-L k-nearest neighbor continuous entropy estimator
       x should be a tensor.
    """
    
    assert k <= x.shape[0] - 1, "Set k smaller than num. samples - 1"
    N, d = x.shape
    intens = 1e-10  # small noise to break degeneracy, see doc.
    x = x + intens*torch.rand(*x.shape, dtype = torch.float).cuda()
    nn = kth_neighbor_distance(x, k, p)
    const = digamma(N) - digamma(k) + d * np.log(2)
    return (const + d * torch.mean(torch.log(nn))) / np.log(base)

def centropy(x, y, k=3, base=2):
    """ The classic K-L k-nearest neighbor continuous entropy estimator for the
        entropy of X conditioned on Y.
    """
    hxy = entropy(cat(x, y), k, base)
    hy = entropy(y, k, base)
    return hxy - hy

def entropyd(xd, base=2):
    """ Discrete entropy estimator
        Given a list of samples which can be any hashable object
    """
    return entropy_p(extract_p(xd)[0], base=base)

def centropyd(xd, yd, base=2):
    """ Discrete entropy estimator for the entropy of X conditioned on Y.
    """
    return entropyd(cat(xd, yd), base) - entropyd(yd, base)

# def mi(x, y, k=3, base=2, p=float('inf')):
#     """ Mutual information of x and y
#         x, y should be a tensor.
#     """
# #     breakpoint()
#     x = x.cuda()
#     y = y.cuda()
#     assert x.shape[0] == y.shape[0], "Lists should have same length"
#     assert k <= x.shape[0] - 1, "Set k smaller than num. samples - 1"
#     intens = 1e-10  # small noise to break degeneracy, see doc.
#     x = x + intens*torch.rand(*x.shape, dtype = torch.float).cuda()
#     y = y + intens*torch.rand(*y.shape, dtype = torch.float).cuda()
#     points = cat(x, y)
#     # Find nearest neighbors in joint space, p=inf means max-norm
#     dvec = kth_neighbor_distance(points, k, p)
#     a, b, c, d = avgdigamma(x, dvec), avgdigamma(y, dvec), digamma(k), digamma(len(x))
#     return (-a - b + c + d) / np.log(base)

def cmi(x, y, z, k=3, base=2):
    """ Mutual information of x and y, conditioned on z
        x, y, z should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
        if x is a one-dimensional scalar and we have four samples
    """
    assert x.shape[0] == y.shape[0], "Lists should have same length"
    assert k <= x.shape[0] - 1, "Set k smaller than num. samples - 1"
    intens = 1e-10  # small noise to break degeneracy, see doc.
    x = x + intens*np.random.rand(*x.shape).cuda()
    y = y + intens*np.random.rand(*y.shape).cuda()
    z = z + intens*np.random.rand(*z.shape).cuda()
    points = cat(x, y, z)
    # Find nearest neighbors in joint space, p=inf means max-norm
    dvec = kth_neighbor_distance(points, k)
    a, b, c, d = avgdigamma(cat(x, z), dvec), avgdigamma(cat(y, z), dvec), avgdigamma(z, dvec), digamma(k)
    return (-a - b + c + d) / np.log(base)

def midd(xd, yd, base=2):
    """ Discrete mutual information estimator
        Given a list of samples which can be any hashable object
    """
    return -entropyd(cat(xd, yd), base) + entropyd(xd, base) + entropyd(yd, base)

def cmidd(xd, yd, zd):
    """ Discrete mutual information estimator
        Given a list of samples which can be any hashable object
    """
    return entropyd(cat(yd, zd)) + entropyd(cat(xd, zd)) - entropyd(cat(xd, yd, zd)) - entropyd(zd)

def micd(xc, yd, k=3, base=2, warning=False):
    """ If x is continuous and y is discrete, compute mutual information
    """
#     overallentropy = entropy(xc, k, base)
    overallentropy = mi(xc, xc, k, base)

    n = yd.shape[0]
    y_unique_array, y_unique_inverse, y_unique_counts = unique(yd, dim = 0)
    py = y_unique_counts/ torch.sum(y_unique_counts, dtype = torch.float)

    MI = overallentropy.clone()
    for i, yval in enumerate(y_unique_array):
        ind = torch.all(yd==yval, dim=-1)
        xgiveny = xc[ind]
        if k <= xgiveny.shape[0] - 1:
#             MI -= py[i] * entropy(xgiveny, k, base)
            MI -= py[i] * mi(xgiveny, xgiveny, k, base)
        else:
            if warning:
                print("Warning, after conditioning, on y=", yval, " insufficient data. Assuming maximal entropy in this case.")
            MI -= py[i] * overallentropy
    return torch.abs(MI)  # units already applied

def midc(yd, xc, k=3, base=2, warning=False):
    return micd(xc, yd, k, base, warning)

def avgdigamma(points, dvec, p=float('inf')):
    # This part finds number of neighbors in some radius in the marginal space
    # returns expectation value of <psi(nx)>
#     breakpoint()
    N = points.shape[0]
    n = 0
    batch_size = BATCHSIZE
    total = 0.
    while n<N:
        all_distance = torch.norm(torch.unsqueeze(points[n:n+batch_size],1)-points, p, -1)
        ball_points_num = torch.sum(all_distance<(torch.unsqueeze(dvec[n:n+batch_size], 1)-1e-15), 1)
        total += torch.sum(torch.digamma(ball_points_num))
        n += batch_size
    return total/N

def cat(*args):
    if isinstance([*args][0], torch.Tensor):
        return torch.cat([*args], dim=1)
    else:
        return np.concatenate([*args], axis=-1)
    
def kth_neighbor_distance(x, k, p=float('inf')):
#     breakpoint()
    N = x.shape[0]
    n = 0
    batch_size = BATCHSIZE
    distance = torch.zeros(N).cuda()
    while n<N:
        all_distance = torch.norm(torch.unsqueeze(x[n:n+batch_size],1)-x, p, -1)
        distance[n:n+batch_size] = torch.kthvalue(all_distance, k+1, -1)[0]
#     assign_index = knn(x, x, k+1)[:, k::(k+1)]
#     vector = x[assign_index[0] - x[assign_index][1]
#     distance = torch.norm(vector, p, dim=-1)
        n += batch_size
    return distance

############### Each Task ###############
def calc_information_sampling(x, y, z, k=3, base=2):
    Ixz = midc(x, z, k, base)
    Izy = micd(z, y, k, base)
    Ixy = midd(x, y, base)
    
    Hx  = entropyd(x, base)
    Hy  = entropyd(y, base)
    Hz  = mi(z, z, k, base)
    return Ixz, Izy, Ixy, Hz, Hx, Hy

@ram_monitor
@time_monitor
def calc_information_from_node(x, y, z, k=3, base=2):
    node_infm = torch.zeros([z.shape[1], 2], dtype = torch.float)
    
    for i in range(z.shape[1]):
        node = z[:,[i]]
#         node_infm[i, 0] = midc(x, node, k, base)
#         node_infm[i, 1] = micd(node, y, k, base)
        node_infm[i, 0] = mi(x, node, k, base)
        node_infm[i, 1] = mi(node, y, k, base)
    return node_infm, torch.sum(node_infm, dim = 0)

@ram_monitor
@time_monitor
def calc_information_diff_nodes(x, y, z, all_combinations, k=3, base=2):
    com_dic = {}
    com_infm = torch.zeros([len(all_combinations), 5], dtype = torch.float)
    
    for idx, com in enumerate(all_combinations):
        node_idx = [i-1 for i in com]
        com_nodes = z[:, node_idx]
        
        Ixz, Izy, _, Hz, Hx, Hy = calc_information_sampling(x, y, com_nodes, k, base)
        com_infm[idx, 0] = Ixz
        com_infm[idx, 1] = Izy
        com_infm[idx, 2] = Hz
        com_infm[idx, 3] = Hx
        com_infm[idx, 4] = Hy
        com_dic[com] = com_infm[idx]
    
    return com_dic

def calc_information_sum_diff_nodes(z, all_combinations, node_infm, com_infm):
    last_node = z.shape[1]
    
    sum_dic    = {}
    node_range = set(range(1, last_node + 1))
    
    for idx, com in enumerate(all_combinations):
        rest_node = tuple(sorted(node_range.difference(set(com))))
        if len(rest_node) > 1:
            sum_diff_nodes = com_infm[com][:2] + com_infm[rest_node][:2]
            sum_dic[com]   = {f"{com} + {rest_node}": sum_diff_nodes}
        else:
            sum_diff_nodes = node_infm[rest_node[0] - 1] + com_infm[com][:2]
            sum_dic[com]   = {f"({rest_node[0]}) + {com}": sum_diff_nodes}
    return sum_dic

def calc_information_between_nodes(z1, z2, k=3, base=2):
#     Hz1 = entropy(z1, k, base)
    Hz1 = mi(z1, z1, k, base)
    Hz1_z2 = centropy(z1, z2, k, base)
    Iz1z2 = mi(z1, z2, k, base)
    
    return Hz1, Hz1_z2, Iz1z2


############### Doing epochs ###############
def calc_informations(x, y, z, all_combinations, k=3, base=2):
    Ixz, Izy, Ixy, Hz, Hx, Hy = calc_information_sampling(x, y, z, k, base)
    params = torch.zeros((1, 6))
    params[0, 0] = Ixz
    params[0, 1] = Izy
    params[0, 2] = Ixy
    params[0, 3] = Hz
    params[0, 4] = Hx
    params[0, 5] = Hy
    
#     node_infm, sum_node_infm = calc_information_from_node(x, y, z, k, base)
#     com_infm = calc_information_diff_nodes(x, y, z, all_combinations, k, base)
    
#     breakpoint()
#     sum_infm = calc_information_sum_diff_nodes(z, all_combinations, node_infm, com_infm)
    
    node_infm, sum_node_infm, com_infm, sum_infm = None, None, None, None
    
    return params, node_infm, sum_node_infm, com_infm, sum_infm

def calc_informations_all_combinations_between_nodes(z, combinations_between_nodes, k=3, base=2):
    com_nodes = {}

    for proportion in combinations_between_nodes:
        pro = list(map(lambda a: list(map(lambda b: b - 1, a)), proportion))
        
        z1 = z[:, pro[0]]
        z2 = z[:, pro[1]]
        
        Hz1, Hz1_z2, Iz1z2 = calc_information_between_nodes(z1, z2, k, base)

        com_nodes[', '.join(str(i) for i in proportion)] = Iz1z2.item()

    return com_nodes

def calc_informations_all_combinations_between_nodes_layer(z, all_combinations, k=3, base=2):
    last_node = z.size(1)
    
    combNode_layer = {}
    for idx, com in enumerate(all_combinations):
        node_idx = [i-1 for i in com]
        nodes = z[:,node_idx]
        combNode_layer[f'{com}'] = calc_information_between_nodes(nodes, z, k, base)[-1].item()
    combNode_layer[f'{tuple(range(1, last_node+1))}'] = calc_information_between_nodes(z, z, k, base)[-1].item()
    
    return combNode_layer

def calc_informations_S_T_single_node(Sz, Tz, k=3, base=2):
    last_node = Sz.size(1)
    
    ST_single_node = {}
    for i in range(last_node):
        S_node, T_node = Sz[:,[i]], Tz[:,[i]]
        ST_single_node[f'({i+1})'] = calc_information_between_nodes(S_node, T_node, k, base)[-1].item()
    
    return ST_single_node

def calc_informations_S_T_com_nodes(Sz, Tz, all_combinations, k=3, base=2):
    last_node = Sz.size(1)
    
    ST_com_nodes = {}
    for idx, com in enumerate(all_combinations):
        node_idx = [i-1 for i in com]
        S_nodes, T_nodes = Sz[:,node_idx], Tz[:,node_idx]
        ST_com_nodes[f'{com}'] = calc_information_between_nodes(S_nodes, T_nodes, k, base)[-1].item()
    ST_com_nodes[f'{tuple(range(1, last_node+1))}'] = calc_information_between_nodes(Sz, Tz, k, base)[-1].item()
    
    return ST_com_nodes
            

############### Call ###############
def knn_get_information(z, x, y, interval_information_display, all_combinations, k=3, base=2):
    z = z.detach()
    params, node_infm, sum_node_infm, com_infm, sum_infm = \
    calc_informations(x, y, z, all_combinations, k, base)

    return params, node_infm, sum_node_infm, com_infm, sum_infm

@ram_monitor
@time_monitor
def knn_get_information_between_nodes(z, combinations_between_nodes, k=3, base=2):
    z = z.detach()
    comb_nodes = calc_informations_all_combinations_between_nodes(z, combinations_between_nodes, k, base)

    return comb_nodes

@ram_monitor
@time_monitor
def knn_get_information_combNode_layer(z, all_combinations, k=3, base=2):
    z = z.detach()
    combNode_layer = calc_informations_all_combinations_between_nodes_layer(z, all_combinations, k, base)
    
    return combNode_layer

@ram_monitor
@time_monitor
def knn_get_information_S_T(Sz, Tz, all_combinations, k=3, base=2):
    Sz = Sz.detach()
    Sz = Sz.detach()
    ST_single_node = calc_informations_S_T_single_node(Sz, Tz, k, base)
    ST_com_nodes = calc_informations_S_T_com_nodes(Sz, Tz, all_combinations, k, base)

    return ST_single_node, ST_com_nodes




if __name__=="__main__":
    import torch
    from torch_cluster import knn
    x = torch.rand(3000,1245).cuda()
    y = torch.randint(1,5,(3000,1)).cuda()
    breakpoint()
    mi(y, y)
    breakpoint()

















