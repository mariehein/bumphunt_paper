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

class MAF:
    def __init__(self, features, conditionals, transforms, hidden, blocks):
        self.features = features
        self.conditionals = conditionals
        self.transforms = transforms
        self.hidden = hidden
        self.blocks = blocks

    def make_MAF(self, device):
        base_dist = nflows.distributions.normal.StandardNormal(shape=[self.features])

        inds = np.arange(self.features)
        list_transforms = []
        for i in range(self.transforms):
            np.random.shuffle(inds)
            list_transforms.append(
                nflows.transforms.permutations.Permutation(torch.tensor(inds))
            )
            list_transforms.append(
                nflows.transforms.autoregressive.MaskedAffineAutoregressiveTransform(
                    features=self.features,
                    hidden_features=self.hidden,
                    context_features=self.conditionals, # dimension of conditions.
                    num_blocks=self.blocks,
                    activation=torch.nn.functional.gelu
                )
            )

        self.transform = nflows.transforms.base.CompositeTransform(list_transforms).to(device)
        self.flow = nflows.flows.base.Flow(self.transform, base_dist).to(device)

    def train(self, train_data, val_data, opt, scheduler, args):
        print('Starting training...')

        n_batches = train_data.shape[0] // args.batch_size

        def train_step(flow, x, context):
            opt.zero_grad()
            loss = - flow.log_prob(x, context=context).mean()
            loss.backward()
            opt.step()
            return loss
        
        loss_best = np.inf
        patience = 0

        train_loss_arr = []
        val_loss_arr = []

        for epoch in range(args.epochs):
            print(torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
            np.random.shuffle(train_data)
            batch_losses = []
            for i in range(n_batches):
                batch = train_data[i * args.batch_size:(i+1) * args.batch_size]
                loss = train_step(self.flow, batch[:,self.conditionals:], context = batch[:,:self.conditionals])
                batch_losses.append(loss.detach().cpu().numpy())
            loss = np.mean(batch_losses)
            val_loss = -self.flow.log_prob(val_data[:,self.conditionals:], context = val_data[:,:self.conditionals]).mean()
            scheduler.step(val_loss)

            train_loss_arr.append(loss)
            val_loss_arr.append(val_loss.detach().cpu().numpy())

            if val_loss < loss_best:
                loss_best = val_loss
                torch.save(self.flow.state_dict(), args.directory+"MAF_state_dict")
                patience=0
            else:
                if patience < args.patience_max:
                    patience+=1
                else:
                    break

            print('Epoch: {}, Loss: {}, Val Loss: {}'.format(epoch, loss, val_loss))
        print('Training finished.')

        history = {"val_loss": torch.tensor(val_loss_arr).cpu().numpy(), "train_loss": torch.tensor(train_loss_arr).cpu().numpy()}
        return history

    
    def sample(self, device, m, N):
        kernel = gaussian_kde(m)
        m_samples = kernel.resample(size=int(N)).T

        with torch.inference_mode():
            tensor_generated = self.flow.sample(
                1 , 
                context=torch.tensor(m_samples, device=device, dtype=torch.float32)
            )
        arr_generated = tensor_generated.detach().cpu().numpy()[:,0,:]
        samples = np.concatenate((m_samples, arr_generated), axis=1)
        return samples

def sample(model, device, m, N, norm, directory, plot=True, testset=None, name="samples"):
    samples = model.sample(device, m, N)
    samples = norm.inverse(samples)
    np.save(directory+name+".npy", samples)

    if plot:
        if testset is None:
            fig, ax = plt.subplots(2,2)
            ax = ax.flatten()
            legend = [None, "samples", None, None]
            for i in range(4):
                ax[i].hist(samples[:,i+1], density=True, bins=50, label = legend[i])
                ax[i].set_yscale('log')
                if i == 1:
                    ax[i].legend()
            fig.savefig(directory+name+".pdf")
        
        else:    
            fig, ax = plt.subplots(2,2)
            ax = ax.flatten()
            legend = [None, ("samples", "bkg"), None, None]
            for i in range(4):
                ax[i].hist((samples[:,i+1], testset[:,i+1]), density=True, bins=50, label = legend[i])
                ax[i].set_yscale('log')
                if i == 1:
                    ax[i].legend()
            fig.savefig(directory+name+".pdf")
        plt.close('all')

def loss_saving(history, directory, plot=True):
    np.save(directory+"losses.npy", history)
    if plot:
        plt.figure()
        plt.plot(history["train_loss"][1:], label="train loss")
        plt.plot(history["val_loss"][1:], label="val loss")
        plt.savefig(directory+"losses.pdf")

def run_MAF(args, train_data, val_data, m_inner, m_outer, test_data, inner, norm):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("INFO: current device: {}".format(device))

    model = MAF(transforms=10, blocks=3, hidden=32, features=args.inputs, conditionals=args.conditional_inputs)
    model.make_MAF(device)
    
    optimizer = torch.optim.Adam(model.flow.parameters(),lr = args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 5)

    tensor_input_train = torch.tensor(train_data, device=device, dtype=torch.float32)
    tensor_input_valid = torch.tensor(val_data, device=device, dtype=torch.float32)

    history = model.train(tensor_input_train, tensor_input_valid, optimizer, scheduler, args)

    loss_saving(history, args.directory)

    sample(model.flow, device, m_inner, args.N_samples, norm, args.directory, testset=inner, name="samples_inner")
    sample(model.flow, device, m_outer, args.N_samples, norm, args.directory, testset=test_data, name="samples_outer")

    torch.save(model.flow, args.directory+"trained_flow.pt")