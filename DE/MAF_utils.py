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
import psutil

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
                    activation=torch.nn.functional.relu
                )
            )
            list_transforms.append(transforms.normalization.BatchNorm(self.features, momentum=1.))

        self.transform = nflows.transforms.base.CompositeTransform(list_transforms).to(device)
        self.flow = nflows.flows.base.Flow(self.transform, base_dist).to(device)

    def train(self, train_data, val_data, opt, scheduler, args, device):
        print('Starting training...')

        n_batches = train_data.shape[0] // args.batch_size

        def train_step(flow, x, context):
            opt.zero_grad()
            loss = - flow.log_prob(x, context=context).mean()
            loss.backward()
            opt.step()
            return loss

        def loss_calculation_batches(flow, val):
            batch_loss = []
            for batch in val:
                a = -flow.log_prob(batch[:,1:], context = batch[:,:1]).mean().detach().cpu().numpy()
                batch_loss.append(a)
            return np.mean(batch_loss)
                
        loss_best = np.inf
        patience = 0

        train_loss_arr = []
        val_loss_arr = []
        for epoch in range(args.epochs):

            batch_losses = []
            for batch in train_data:
                loss = train_step(self.flow, batch[:,self.conditionals:], context = batch[:,:self.conditionals])
                batch_losses.append(loss.detach().cpu().numpy())
            loss = np.mean(batch_losses)

            val_loss = loss_calculation_batches(self.flow, val_data)
            #scheduler.step(val_loss)

            train_loss_arr.append(loss)
            val_loss_arr.append(val_loss)

            if val_loss < loss_best:
                loss_best = val_loss
                torch.save(self.flow.state_dict(), args.directory+"MAF_state_dict")
                patience=0
            else:
                if patience < args.patience_max:
                    patience+=1
                else:
                    break

            if args.save_models:
                torch.save(self.flow.state_dict(), args.directory+"epoch"+str(epoch)+"_dict")

            print('Epoch: {}, Loss: {}, Val Loss: {}'.format(epoch, loss, val_loss), flush=True)

            del loss, val_loss, batch, batch_losses

        print('Training finished.')

        self.flow.load_state_dict(torch.load(args.directory+"MAF_state_dict", map_location=device))

        history = {"val_loss": np.array(val_loss_arr), "train_loss": np.array(train_loss_arr)}
        return history

    
    def sample(self, device, m, N):
        kernel = gaussian_kde(m)
        m_samples = kernel.resample(size=int(N)).T
        print(m_samples.shape)

        with torch.inference_mode():
            tensor_generated = self.flow.sample(
                1 , 
                context=torch.tensor(m_samples, device=device, dtype=torch.float32)
            )
        arr_generated = tensor_generated.detach().cpu().numpy()[:,0,:]
        samples = np.concatenate((m_samples, arr_generated), axis=1)
        print(samples.shape)
        return samples

def sample(model, device, m, N, norm, directory, plot=True, testset=None, name="samples"):
    samples = model.sample(device, m, N)
    print(samples)
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

    print(psutil.Process().memory_info().rss / (1024 * 1024))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("INFO: current device: {}".format(device))

    model = MAF(transforms=args.transforms, blocks=args.blocks, hidden=args.hidden, features=args.inputs, conditionals=args.conditional_inputs)
    model.make_MAF(device)

    print(psutil.Process().memory_info().rss / (1024 * 1024))
    
    optimizer = torch.optim.Adam(model.flow.parameters(),lr = args.learning_rate, weight_decay = 0.000001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 2)

    print(psutil.Process().memory_info().rss / (1024 * 1024))
    tensor_input_train = torch.tensor(train_data, device=device, dtype=torch.float32)
    tensor_input_valid = torch.tensor(val_data, device=device, dtype=torch.float32)

    train_loader = torch.utils.data.DataLoader(tensor_input_train, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(tensor_input_valid, batch_size=args.batch_size, shuffle=True)

    print(psutil.Process().memory_info().rss / (1024 * 1024))
    history = model.train(train_loader, val_loader, optimizer, scheduler, args, device)

    loss_saving(history, args.directory)

    model.flow.eval()

    print(psutil.Process().memory_info().rss / (1024 * 1024))
    sample(model, device, m_inner, args.N_samples, norm, args.directory, testset=inner, name="samples_inner")
    sample(model, device, m_outer, args.N_samples, norm, args.directory, testset=test_data, name="samples_outer")

    print(psutil.Process().memory_info().rss / (1024 * 1024))
    torch.save(model.flow, args.directory+"trained_flow.pt")