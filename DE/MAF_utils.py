import torch
import nflows
from nflows import flows
from nflows import transforms
from nflows import distributions
from matplotlib import pyplot as plt
import time
import math
import numpy as np
from scipy.stats import gaussian_kde

def run_MAF(args, train_data, val_data, m_inner, m_outer, test_data, inner, norm):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("INFO: current device: {}".format(device))

    num_transforms = 10
    base_dist = nflows.distributions.normal.StandardNormal(shape=[4])

    permutations = [[2,1,3,0], [1,3,0,2], [0,2,1,3], [3,0,2,1], [2,1,3,0], [1,3,0,2], [0,2,1,3], [3,0,2,1], [2,1,3,0], [1,3,0,2]]
    list_transforms = []
    for i in range(num_transforms):
        list_transforms.append(
            nflows.transforms.permutations.Permutation(torch.tensor(permutations[i]))
        )
        list_transforms.append(
            nflows.transforms.autoregressive.MaskedAffineAutoregressiveTransform(
                features=4,
                hidden_features=32,
                context_features=1, # dimension of conditions.
                num_blocks=3,
                activation=torch.nn.functional.gelu
            )
        )

    transform = nflows.transforms.base.CompositeTransform(list_transforms).to(device)

    flow1 = nflows.flows.base.Flow(transform, base_dist).to(device)

    optimizer = torch.optim.Adam(flow1.parameters(),lr=0.01)

    num_iter = 10000
    time_start = time.time()

    loss_best = np.inf

    patience=0
    patience_max=500

    tensor_input_train = torch.tensor(train_data[:,1:], device=device, dtype=torch.float32)
    tensor_input_valid = torch.tensor(val_data[:,1:], device=device, dtype=torch.float32)
    tensor_context_train = torch.tensor(train_data[:,:1], device=device, dtype=torch.float32)
    tensor_context_valid = torch.tensor(val_data[:,:1], device=device, dtype=torch.float32)

    for epoch in range(num_iter):
        optimizer.zero_grad()

        loss = -flow1.log_prob(inputs=tensor_input_train, context=tensor_context_train).mean()
        loss.backward()

        with torch.no_grad():
            loss_valid = -flow1.log_prob(inputs=tensor_input_valid, context=tensor_context_valid).mean().detach().cpu().numpy()

        if loss_best > loss_valid:
            print("epoch {:04d}: loss improved {:.4f} -> {:.4f}".format(epoch, loss_best, loss_valid))
            loss_best = loss_valid
            torch.save(flow1.state_dict(), "MAF_state_dict")
            patience=0
        else:
            if patience < patience_max:
                patience += 1
            else:
                break

        optimizer.step()

    if (epoch+1) % 250 == 0:
        time_current = time.time()
        print("INFO: epoch {:04d} | time_elapsed: {:.3f}s".format(epoch+1, time_current - time_start))
        print("INFO:            | loss: {:.4f} | loss_best: {:.4f}".format(loss, loss_best))
    print("INFO: training finished")

    flow1.load_state_dict(torch.load("MAF_state_dict", map_location=device))

    kernel_inner = gaussian_kde(m_inner)
    m_inner_samples = kernel_inner.resample(size=int(1e6)).T

    with torch.inference_mode():
        tensor_generated = flow1.sample(
            1, # number of samples in context (nflows convention....)
            context=torch.tensor(m_inner_samples, device=device, dtype=torch.float32)
        )
        arr_generated = tensor_generated.detach().cpu().numpy()[:,0,:]
    print(arr_generated.shape)
    samples_inner = norm.inverse(np.concatenate(m_inner_samples, arr_generated))
    np.save(args.directory+"samples_inner.npy", samples_inner)

    fig, ax = plt.subplots((2,2))
    ax = ax.flatten()
    for i in range(4):
        ax[i].hist((samples_inner[:,i+1], inner[:,i+1]))
    fig.savefig(args.directory+"inner_samples.pdf")

    kernel_outer = gaussian_kde(m_outer)
    m_outer_samples = kernel_outer.resample(size=int(1e6)).T

    with torch.inference_mode():
        tensor_generated = flow1.sample(
            1, # number of samples in context (nflows convention....)
            context=torch.tensor(m_outer_samples, device=device, dtype=torch.float32)
        )
        arr_generated = tensor_generated.detach().cpu().numpy()[:,0,:]
    print(arr_generated.shape)
    samples_outer = norm.inverse(np.concatenate(m_outer_samples, arr_generated))
    np.save(args.directory+"samples_outer.npy", samples_outer)

    fig, ax = plt.subplots((2,2))
    ax = ax.flatten()
    for i in range(4):
        ax[i].hist((samples_outer[:,i+1], test_data[:,i+1]))
    fig.savefig(args.directory+"outer_samples.pdf")