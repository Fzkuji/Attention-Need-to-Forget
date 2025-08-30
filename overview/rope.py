# https://zhuanlan.zhihu.com/p/692197097
# 感谢沐清宇的笔记

import torch
import matplotlib.pyplot as plt
import numpy as np

def multi_ploat(x):
    x_np = x.numpy()
    theta_list = [5000, 10000, 20000, 50000, 100000, 500000, 1000000, 5000000, 10000000]
    n = [i for i in range(len(x_np[0]))]
    fig = plt.figure()
    fig.patch.set_alpha(0)
    for i in range(len(x_np)):
        plt.plot(n, x_np[i])
    ax = plt.gca()
    ax.patch.set_alpha(0)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='gray')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.savefig('plot.pdf', format='pdf', bbox_inches='tight', transparent=True)
    plt.show()


def my_print(x):
    print(x.shape, x)

def precompute_freqs_cis(dim=128, end=4096, theta=10000.0):
    print(f"dim = {dim}, end={end}, theta={theta}")
    range_arr = torch.arange(0, dim, 2)[:(dim // 2)].float() / dim
    freqs = 1 / (theta ** range_arr)
    t = torch.arange(end) # tensor([   0,    1,    2,  ..., 4093, 4094, 4095])
    # ploat(freqs)
    freqs = torch.outer(t, freqs).float() # 从64，扩展到 [4096, 64]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) # 把长度为1，角度为freqs从极坐标转换为复数tensor
    return freqs_cis, freqs

xq = torch.view_as_complex(torch.ones([1, 1, 1, 128]).float().reshape(1, -1, 2))
xk = torch.view_as_complex(torch.ones([1, 1, 1, 128]).float().reshape(1, -1, 2))


idx = [i for i in range(2000)]

def get_diff_cosine_theta(theta):
    _, freqs = precompute_freqs_cis(dim=128, end=200000, theta=theta)
    cos_freqs = torch.cos(freqs).transpose(1, 0)
    return cos_freqs[1]


def get_scores(theta):
    all_res = []
    freqs_cis, _ = precompute_freqs_cis(dim=128, end=200000, theta=theta)
    for i in idx:
        xq_pos = torch.view_as_real(xq * freqs_cis[0]).flatten(1)
        xk_pos = torch.view_as_real(xk * freqs_cis[i]).flatten(1)
        score = (xq_pos @ xk_pos.T).numpy()[0][0]
        all_res.append(score)
    print(f"theta = {theta}")
    print(all_res)
    return all_res

theta_list = [10000, 100000, 1000000, 10000000]
get_diff_cosine_theta(10000)
total_theta_score = []
total_theta_cos = []
for one_theta in theta_list:
    one_scores = get_scores(one_theta)
    one_cos = get_diff_cosine_theta(one_theta)
    total_theta_score.append(one_scores)
total_theta_score_tensor = torch.from_numpy(np.array(total_theta_score))
multi_ploat(total_theta_score_tensor)

# freqs_cis_reshape = freqs_cis.view()