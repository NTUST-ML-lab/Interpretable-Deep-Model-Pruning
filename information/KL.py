import torch
import torch.nn.functional as F
from information.information_process_bins import extract_p

def KLdivergence(P, Q, reduction="batchmean", device = "cuda"):
    '''
    P: target，目標分佈P
    Q: input，待度量分佈Q
    '''
    return F.kl_div(Q.view(-1).to(device).softmax(-1).log(), P.view(-1).to(device).softmax(-1), reduction=reduction).detach()

def calculate_probability(unique_inverse_z):
    unique_values, counts = torch.unique_consecutive(unique_inverse_z, return_counts=True)
    total_counts = counts.sum()
    probabilities = counts.float() / total_counts
    return probabilities

def KLdivergence_ds(P, Q, BINS, reduction="batchmean", device="cuda"):
    '''
    P: target，目標分佈P
    Q: input，待度量分佈Q
    '''
    digitized_p = BINS[torch.bucketize(P, BINS) - 1]
    digitized_q = BINS[torch.bucketize(Q, BINS) - 1]
    pp, unique_inverse_p = extract_p(digitized_p)
    pp = pp[unique_inverse_p]
    pp /= pp.sum()
    pq, unique_inverse_q = extract_p(digitized_q)
    pq = pq[unique_inverse_q] 
    pq /= pq.sum()
    return torch.sum(pp * torch.log(pp / pq))

if __name__ == "__main__":
    import time
    p = torch.randn( (256, 3) )
    q = torch.randn( (256, 3) )

    a = time.time()
    print(KLdivergence(p, q))
    b = time.time()
    print(b-a)
    print(KLdivergence(q, p))
    pass