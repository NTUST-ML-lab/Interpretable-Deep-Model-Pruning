import torch
import time
import datetime

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
def unique(digitized, dim = None):
    unique_array, unique_inverse, counts = torch.unique(digitized, sorted = True, 
                                                  return_inverse = True, 
                                                  return_counts = True, dim = dim)
    perm = torch.arange(unique_inverse.size(0), 
                        dtype = unique_inverse.dtype,
                        device = unique_inverse.device)
    
    inverse, perm = unique_inverse.flip([0]), perm.flip([0])
    
    return unique_array, inverse.new_empty(unique_array.size(dim)).scatter_(0, inverse, perm), unique_inverse, counts

def extract_p(x):
    unique_array, unique_indices, unique_inverse, unique_counts = unique(x, dim = 0)
    p = torch.divide(unique_counts, torch.sum(unique_counts, dtype = torch.float64))
    
    return p, unique_inverse

def entropy_p(p):
    return -torch.sum(p * torch.log2(p))

def condtion_entropy(py, x, unique_inverse_y):
    Hx_y = 0
    for i in range(py.shape[0]):
        current_xs    = x[unique_inverse_y == i, :]
        _, _, _, unique_counts = unique(current_xs, dim = 0)

        p_current_xs  = unique_counts / torch.sum(unique_counts, dtype = torch.float64)
        Hx_y += (py[i]*(-torch.sum(p_current_xs * torch.log2(p_current_xs))))
    return Hx_y

def mutual_information(x, px, py, unique_inverse_y):
    Hx = entropy_p(px)
    Hx_y = condtion_entropy(py, x, unique_inverse_y)
    Ixy = Hx - Hx_y
    return Hx, Hx_y, Ixy

def mutual_informations(z, x, y, px, py, unique_inverse_x, unique_inverse_y, l_mask=None):
    pz, unique_inverse_z = extract_p(z)
    Hz = entropy_p(pz)
    Hz_x = condtion_entropy(px, z, unique_inverse_x)
    Ixz = Hz - Hz_x
    
    Hx = entropy_p(px)
    if l_mask!=None:
        z = z[l_mask]
        x = x[l_mask]
        pz, unique_inverse_z = extract_p(z)
        px, unique_inverse_x = extract_p(x)
        Hz = entropy_p(pz)
        Hx = entropy_p(px)
    Hz_y = condtion_entropy(py, z, unique_inverse_y)
    Izy = Hz - Hz_y
    
    Hy = entropy_p(py)
    Hy_x = condtion_entropy(px, y, unique_inverse_x)
    Ixy = Hy - Hy_x

    return Ixz, Izy, Ixy, Hz, Hx, Hy

############### Each Task ###############
def calc_information_sampling(z, x, y, px, py, unique_inverse_x, unique_inverse_y, l_mask=None):
    Ixz, Izy, Ixy, Hz, Hx, Hy = mutual_informations(z, x, y, px, py, unique_inverse_x, unique_inverse_y, l_mask)
    
    return Ixz, Izy, Ixy, Hz, Hx, Hy

@ram_monitor
@time_monitor
def calc_information_from_node(z, x, y, px, py, unique_inverse_x, unique_inverse_y, l_mask=None):
    node_infm = torch.zeros([z.shape[1], 2], dtype = torch.float64)
    
    for i in range(z.shape[1]):
        nodes = z[:,[i]]
#         pnodes, unique_inverse_nodes = extract_p(nodes)
        node_infm[i, 0], node_infm[i, 1], _, _, _, _ = mutual_informations(nodes, x, y, px, py,
                                                                           unique_inverse_x, unique_inverse_y, l_mask)
    return node_infm

@ram_monitor
@time_monitor
def calc_information_diff_nodes(z, x, y, px, py, unique_inverse_x, unique_inverse_y, all_combinations, l_mask=None):
    com_dic = {}
    com_infm = torch.zeros([len(all_combinations), 5], dtype = torch.float64)
    
    for idx, com in enumerate(all_combinations):
        node_idx = [i-1 for i in com]
        com_nodes = z[:, node_idx]
        
        Ixz, Izy, _, Hz, Hx, Hy = calc_information_sampling(com_nodes, x, y, px, py,
                                                            unique_inverse_x, unique_inverse_y, l_mask)
        com_infm[idx, 0] = Ixz
        com_infm[idx, 1] = Izy
        com_infm[idx, 2] = Hz
        com_infm[idx, 3] = Hx
        com_infm[idx, 4] = Hy
        com_dic[com] = com_infm[idx]
    
    return com_dic

def calc_information_sum_diff_nodes(z, all_combinations, node_infm, com_infm):
    last_node = z.size(1)
    
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

def calc_information_between_nodes(z1, z2):
    pz1, unique_inverse_z1 = extract_p(z1)
    pz2, unique_inverse_z2 = extract_p(z2)
    
    Hz1, Hz1_z2, Iz1z2 = mutual_information(z1, pz1, pz2, unique_inverse_z2)
    return Hz1, Hz1_z2, Iz1z2


############### Doing epochs ###############
def calc_informations(z, x, y, px, py, unique_inverse_x, unique_inverse_y, all_combinations, l_mask):
    Ixz, Izy, Ixy, Hz, Hx, Hy = calc_information_sampling(z, x, y, px, py, unique_inverse_x, unique_inverse_y, l_mask)
    params = torch.zeros((1, 6), dtype=torch.float64)
    params[0, 0] = Ixz
    params[0, 1] = Izy
    params[0, 2] = Ixy
    params[0, 3] = Hz
    params[0, 4] = Hx
    params[0, 5] = Hy
    
#     node_infm, node_infm_time, node_infm_usage = calc_information_from_node(z, x, y, px, py, unique_inverse_x, unique_inverse_y, l_mask)
#     com_infm, com_infm_time, com_infm_usage = calc_information_diff_nodes(z, x, y, px, py, unique_inverse_x, unique_inverse_y, all_combinations, l_mask)
#     sum_infm = calc_information_sum_diff_nodes(z, all_combinations, node_infm, com_infm)

#     sum_node_infm, sum_infm = None, None
    
    node_infm, sum_node_infm, com_infm, sum_infm = None, None, None, None
    
    return params, node_infm, sum_node_infm, com_infm, sum_infm

def calc_informations_all_combinations_between_nodes(z, combinations_between_nodes):
    com_nodes = {}
    for proportion in combinations_between_nodes:
        pro = list(map(lambda a: list(map(lambda b: b - 1, a)), proportion))
        
        z1 = z[:, [pro[0]]]
        z2 = z[:, [pro[1]]]
        
        Hz1, Hz1_z2, Iz1z2 = calc_information_between_nodes(z1, z2)

        com_nodes[', '.join(str(i) for i in proportion)] = Iz1z2

    return com_nodes

def calc_informations_all_combinations_between_nodes_layer(z, all_combinations):
    last_node = z.size(1)
    
    combNode_layer = {}
    for idx, com in enumerate(all_combinations):
        node_idx = [i-1 for i in com]
        nodes = z[:,node_idx]
        combNode_layer[f'{com}'] = calc_information_between_nodes(nodes, z)[-1]
    combNode_layer[f'{tuple(range(1, last_node+1))}'] = calc_information_between_nodes(z, z)[-1]
    
    return combNode_layer

def calc_informations_S_T_single_node(Sz, Tz):
    last_node = Sz.size(1)
    
    ST_single_node = {}
    for i in range(last_node):
        S_node, T_node = Sz[:,[i]], Tz[:,[i]]
        ST_single_node[f'({i+1})'] = calc_information_between_nodes(S_node, T_node)[-1].item()
    
    return ST_single_node

def calc_informations_S_T_com_nodes(Sz, Tz, all_combinations):
    last_node = Sz.size(1)
    
    ST_com_nodes = {}
    for idx, com in enumerate(all_combinations):
        node_idx = [i-1 for i in com]
        S_nodes, T_nodes = Sz[:,node_idx], Tz[:,node_idx]
        ST_com_nodes[f'{com}'] = calc_information_between_nodes(S_nodes, T_nodes)[-1].item()
    ST_com_nodes[f'{tuple(range(1, last_node+1))}'] = calc_information_between_nodes(Sz, Tz)[-1].item()
    
    return ST_com_nodes
            

############### Call ###############
def get_information(z, x, y, bins, interval_information_display, all_combinations=None, l_mask=None):
    digitized_z = bins[torch.bucketize(z, bins) - 1]
    
    px, unique_inverse_x = extract_p(x)
    py, unique_inverse_y = extract_p(y)
    
    params, node_infm, sum_node_infm, com_infm, sum_infm = \
    calc_informations(digitized_z, x, y, px, py, unique_inverse_x, unique_inverse_y, all_combinations, l_mask)

    return params, node_infm, sum_node_infm, com_infm, sum_infm

@ram_monitor
@time_monitor
def get_information_between_nodes(z, bins, combinations_between_nodes):
    digitized_z = bins[torch.bucketize(z, bins) - 1]

    comb_nodes = calc_informations_all_combinations_between_nodes(digitized_z, combinations_between_nodes)
    return comb_nodes

@ram_monitor
@time_monitor
def get_information_combNode_layer(z, bins, all_combinations):
    digitized_z = bins[torch.bucketize(z, bins) - 1]
    
    combNode_layer = calc_informations_all_combinations_between_nodes_layer(digitized_z, all_combinations)
    
    return combNode_layer

@ram_monitor
@time_monitor
def get_information_S_T(Sz, Tz, bins, all_combinations):
    digitized_Sz = bins[torch.bucketize(Sz, bins) - 1]
    digitized_Tz = bins[torch.bucketize(Tz, bins) - 1]
    
    ST_single_node = calc_informations_S_T_single_node(digitized_Sz, digitized_Tz)
    ST_com_nodes = calc_informations_S_T_com_nodes(digitized_Sz, digitized_Tz, all_combinations)

    return ST_single_node, ST_com_nodes

def Total_Correlation(Z, bins):
    # [Summation H(zj)] - H(Z)
    device = Z.device
    data_num, node_num = Z.shape
    
    digitized_Z = bins[torch.bucketize(Z, bins) - 1]
    unique_prob_Z, unique_inverse_Z = extract_p(digitized_Z)
    H_Z = entropy_p(unique_prob_Z)
    
    H_zj = torch.zeros([node_num], dtype=torch.float64).to(device)
    for j in range(node_num):
        digitized_zj = digitized_Z[...,j]
        unique_prob_zj, unique_inverse_zj = extract_p(digitized_zj)
        H_zj[j] = entropy_p(unique_prob_zj)
        
    return H_zj.sum() - H_Z