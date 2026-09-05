from torch import lgamma
import torch.nn.functional as F
import torch


def variational_encoder_decoder(layers, input_data):
    l1, l2, l3, l4, alpha_layers, l3_ ,l2_, l1_, rec = layers
    # We give batch_size x original_features_dim
    out = F.tanh(l1(input_data))
    out = F.tanh(l2(out))
    out = F.tanh(l3(out))
    ####### Variational part
    concentration_params = []
    K = len(alpha_layers)
    for k in range(K):
        scale = (k + 1) * 10
        layer = alpha_layers[k]
        a = layer(out)  # scale * (1 + F.relu(layer(h1)))  # this makes each k (group) in the prod with different level of values
        concentration_params.append(a)
    size = list(concentration_params[0].size())
    x, y = size[0], size[1]
    em_normalized = torch.zeros(x, y, K, device=input_data.device)
    alphas = torch.stack(concentration_params)  # 10 * (1 + F.relu(torch.stack(concentration_params)))

    alphas_pos = []

    for j in range(x):
        for k in range(K):
            alpha_positive = F.sigmoid(alphas[k, j, :])
            alphas_pos.append(alpha_positive)
            u = torch.rand(alpha_positive.shape, device=input_data.device)
            temp = u.mul(alpha_positive).mul(torch.exp(lgamma(alpha_positive)))
            em = torch.pow(temp, torch.pow(alpha_positive, -1))
            #### the below has the same value as F.softmax(em) but we do below since we need em to be distributed as Dir
            tempreture = torch.pow(em.sum(), -1)
            em = em / em.sum()
            num = em / tempreture
            soft = torch.exp(num) / (torch.exp(num)).sum()
            em_normalized[j, :, k] = soft
            ####

    em_final = torch.flatten(em_normalized, start_dim=1)
    alphas = torch.stack(alphas_pos)

    #######
    # We get batch_size x embedding_dim
    out = F.tanh(l3_(em_final))
    out = F.tanh(l2_(out))
    if l1_.in_features == out.shape[1]:
        out = F.tanh(l1_(out))
    # We get batch_size x original_features_dim
    reconstructed = torch.sigmoid(rec(out))
    return em_final, alphas.view(K, x, y), reconstructed