import argparse
import sys
import os
import shutil
import time
import numpy as np
from scipy.stats import norm

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".cache", "matplotlib"))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.RNN_Class import *



parser = argparse.ArgumentParser(description='PyTorch Elman BPTT Training')
parser.add_argument('--epochs', default=50000, type=int, metavar='N',help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=200, type=int,metavar='N', help='mini-batch size (default: 200)')
parser.add_argument('-o', '--one-sample', default=1, type=int, help='whether one traversal is one sample')
parser.add_argument('-a', '--adam', default=0, type=int, help='whether to use ADAM or not')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,metavar='LR', help='initial learning rate')
parser.add_argument('--lr_step', default='', type=str, help='decreasing strategy')
parser.add_argument('-p', '--print-freq', default=1000, type=int,metavar='N', help='print frequency (default: 1000)')
# parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',help='evaluate model on remapping scenario')
parser.add_argument('-g', '--gpu', default=1, type=int, help='whether enable GPU computing')
parser.add_argument('-n', '--n', default=200, type=int, help='Input/output size')
parser.add_argument('--hidden-n', default=200, type=int, help='Hidden dimension size')
parser.add_argument('-t','--total-steps', default=2000, type=int, help='Total steps per traversal')
parser.add_argument('--savename', default='net', type = str, help='Default output saving name')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--resume-plain', default=0, type=int, help='resume checkpoint without remapping/shuffling input and target')
parser.add_argument('--ae', default=0, type=int, help='Autoencoder or not')
parser.add_argument('--input_osci', default=0, type=int, help='Use oscilatory signal')
parser.add_argument('--noisy', default=0, type=int, help='Gaussian noise sd (Percentage of input)')
parser.add_argument('--noisy2', default=0, type=int, help='whether per element std')
parser.add_argument('--partial', default=0, type=float, help='sparsity level (0-1) amount of the partially trained parameter')
parser.add_argument('--sparsity', default=0, type=float, help='Percentage of active cells in the input layer')
parser.add_argument('--nonoverlap', default=0, type = int, help='If input and trainable weights are nonoverlapping subsets (resume=True)')
parser.add_argument('--input', default='', type=str, help='Load in user defined input sequence')
# parser.add_argument('--recordhidden', default=0, type=int, help='whether to record hidden state every step')
parser.add_argument('--continuous', default=0, type=int, help='whether to inherent hidden state from previous epoch')
parser.add_argument('--relu', default=0, type=int, help='relu activation function in the hidden layer')
parser.add_argument('--interleaved', default=0, type=int, help='train one sample, update weight, train another, update weight')
parser.add_argument('--interval', default=0, type=int, help='the interval of interleaved training')
parser.add_argument('--fixi', default=0, type=int, help='whether fix the input matrix')
parser.add_argument('--fixw', default=0, type=int, help='whether fix the recurrent weight (plus bias)')
parser.add_argument('--nobias', default=0, type=int, help='whether to remove all bias term in RNN module')
parser.add_argument('--custom', default=0, type=float, help='self-defined RNN: hidden dropout probability')
parser.add_argument('--ac_output', default='', type=str, help='set the output activation function to tanh (default softmax)')
parser.add_argument('--Hregularized',default=0,type=float,help='regularization weight (/hidden_N) of the loss function')
parser.add_argument('--noisytrain', default=0, type=int, help='Stochastic input during training, sd=args.noisy')
parser.add_argument('--pred',default=0, type=int, help='whether use one-step future pred loss')
parser.add_argument('--pred2',default=0, type=int, help='whether use multi-step future pred loss, update at each time step')
parser.add_argument('--predfd',default=0, type=int, help='one-step future pred loss with feedback')
parser.add_argument('--pred_d',default=0,type=int,help='multiple-step ahead prediction')
parser.add_argument('--snn', default=0, type=int, help='replace RNN hidden dynamics with LIF spiking dynamics')
parser.add_argument('--auto-snn-tune', default=1, type=int, help='auto-apply stable defaults for SNN (Adam/lr/grad-clip)')
parser.add_argument('--snn-standard', default=1, type=int, help='enable standard DL-SNN defaults (Poisson input + logits readout + CE)')
parser.add_argument('--grad-clip', default=0.0, type=float, help='global grad-norm clipping value (0 disables)')
parser.add_argument('--lif-alpha', default=0.9, type=float, help='synaptic current decay factor')
parser.add_argument('--lif-beta', default=0.9, type=float, help='LIF leak factor')
parser.add_argument('--lif-threshold', default=1.0, type=float, help='LIF spike threshold')
parser.add_argument('--lif-reset', default=1.0, type=float, help='LIF reset amount after spike')
parser.add_argument('--sg-beta', default=10.0, type=float, help='surrogate gradient slope for spike function')
parser.add_argument('--lif-refractory', default=1, type=int, help='absolute refractory period (in steps)')
parser.add_argument('--lif-learn-threshold', default=1, type=int, help='learnable threshold in SNN mode')
parser.add_argument('--input-spike-mode', default='poisson', type=str, help='input encoding in SNN mode: poisson/onoff/analog')
parser.add_argument('--input-spike-scale', default=5.0, type=float, help='input-to-spike scaling factor')
parser.add_argument('--snn-readout', default='softmax_seq', type=str, help='SNN readout: softmax_seq/logits_seq')
parser.add_argument('--snn-loss', default='mse', type=str, help='task loss in SNN mode: mse/ce')
parser.add_argument('--snn-label-source', default='last', type=str, help='label extraction for CE: last/mean/max')
parser.add_argument('--snn-rate-lambda', default=1e-4, type=float, help='spike-rate sparsity regularization weight')
parser.add_argument('--snn-target-rate', default=0.05, type=float, help='target firing rate for spike regularization')
parser.add_argument('--seed', default=-1, type=int, help='random seed for reproducibility (-1 disables)')
# parser.add_argument('--ownnet',default=0, type = float, help='user defined RNN')
# parser.add_argument('--momentum', default=0.01, type=float, metavar='M',
                    # help='momentum')
# parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    # metavar='W', help='weight decay (default: 1e-4)')
# parser.add_argument('--lr_step', default='40,60', help='decreasing strategy')
# parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    # help='use pre-trained model')
# parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    # help='manual epoch number (useful on restarts)')

def torch_load_compat(path, map_location=None):
    """Support old checkpoints on PyTorch>=2.6 where weights_only defaults to True."""
    try:
        if map_location is None:
            return torch.load(path, weights_only=False)
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        if map_location is None:
            return torch.load(path)
        return torch.load(path, map_location=map_location)


def main():
    global args

    args = parser.parse_args()
    if args.seed >= 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    if args.snn and args.auto_snn_tune:
        if args.lr == parser.get_default('lr'):
            args.lr = 0.001
        if args.grad_clip <= 0:
            args.grad_clip = 1.0
        if args.adam == 0:
            args.adam = 1
        if args.snn_rate_lambda < 0:
            args.snn_rate_lambda = 0.0
    if args.snn and args.snn_standard:
        if args.input_spike_mode == parser.get_default('input-spike-mode'):
            args.input_spike_mode = 'poisson'
        if args.snn_readout == parser.get_default('snn-readout'):
            args.snn_readout = 'logits_seq'
        if args.snn_loss == parser.get_default('snn-loss'):
            args.snn_loss = 'ce'
    lr = args.lr
    n_epochs = args.epochs
    RecordEp = args.print_freq
    SeqN = args.batch_size
    N = args.n
    hidden_N = args.hidden_n
    TotalSteps = args.total_steps
    input_payload = None
    inferred_n_from_input = None
    if args.input:
        input_payload = torch_load_compat(args.input)
        if 'X_mini' not in input_payload or 'Target_mini' not in input_payload:
            raise KeyError("Input file must contain 'X_mini' and 'Target_mini'.")
        inferred_n_from_input = int(input_payload['X_mini'].shape[2])
        N = inferred_n_from_input

    global f
    f = open(args.savename+'.txt','w')
    print('Settings:', file = f)
    print(str(sys.argv), file = f)
    if args.input:
        print('Use user-defined input: {}'.format(args.input), file=f)
        print('Override feature size N from --input to {:d}'.format(N), file=f)


    if args.auto_snn_tune and args.snn:
        print(
            'Auto SNN tune enabled: optimizer={}, lr={}, grad_clip={}, rate_lambda={}, standard={}, readout={}, loss={}'.format(
                'Adam' if args.adam else 'SGD',
                args.lr,
                args.grad_clip,
                args.snn_rate_lambda,
                bool(args.snn_standard),
                args.snn_readout,
                args.snn_loss,
            ),
            file=f,
        )

    ## Generate network input unless a prebuilt sequence is provided.
    if args.input:
        X_mini = input_payload['X_mini']
        Target_mini = input_payload['Target_mini']
        print('Loaded input tensor shape: {}'.format(tuple(X_mini.shape)), file=f)
    else:
        # Circular input
        X,Target = BellShape_input(N,TotalSteps)
        if args.input_osci:
            X,Target = Cos_input(N,TotalSteps,args.input_osci)
        # sparse input signal
        if args.sparsity:
            print('sparsity of input: {}'.format(args.sparsity),file=f)
            N_pre = np.int64(N*args.sparsity)
            X_pre,tmp = BellShape_input(N_pre,TotalSteps)
            np.random.seed(2); idx_active1 = np.random.choice(N,N_pre)
            np.random.seed(3); idx_active2 = np.random.choice(N,N_pre)
            X = np.zeros((N,TotalSteps)); Target = np.zeros((N,TotalSteps))
            X[idx_active1,:] = X_pre; Target[idx_active2,:] = X_pre
        # noisy input
        if args.noisy:
            if args.noisy2:
                print('Noisy input ({:d}% of each element)'.format(args.noisy),file=f)
                X = X + args.noisy/100*X*np.random.normal(0,1,X.shape)
            else:
                print('Noisy input ({:d}% of maximum)'.format(args.noisy),file=f)
                sd = args.noisy*np.max(X)/100
                X = X + np.random.normal(0,1,X.shape)*sd
                X = X / np.amax(abs(X))
        if args.noisytrain:
            print('Noisy input ({:d}%), Stochastic input and output during training'.format(args.noisy),file=f)

        # Prepare input and target for the model: decrease the time resolution
        if args.one_sample:
            print('One travesal as one sample: SeqN={:d}'.format(SeqN), file=f)
            Select_T = np.arange(0,TotalSteps,np.int64(TotalSteps/SeqN),dtype=int)
            tmp = np.expand_dims((X[:,Select_T].T),axis=0)
            X_mini = torch.tensor(tmp.astype(np.single))
            tmp = np.expand_dims((Target[:,Select_T].T),axis=0)
            Target_mini = torch.tensor(tmp.astype(np.single)) # Output: (batch*seq*feature)
        else:
            print('Splitting into multiple samples', file=f)
            b_idx = np.arange(0,X.shape[1],SeqN)
            X_batch = np.zeros((b_idx.shape[0],SeqN,np.int64(N))) #NBatch * NSeq * NFeature
            Target_batch = np.zeros((b_idx.shape[0],SeqN,np.int64(N)))
            for i in range(b_idx.shape[0]):
                X_batch[i,:,:] = X[:,b_idx[i]:b_idx[i]+SeqN].T
                Target_batch[i,:,:] = Target[:,b_idx[i]:b_idx[i]+SeqN].T
            X_mini = torch.tensor(X_batch.astype(np.single))
            Target_mini = torch.tensor(Target_batch.astype(np.single))

    ##  define network module
    net = ElmanRNN_pytorch_module(N,hidden_N,N)
    if args.ac_output == 'none':
        print('Remove the output activation function', file = f)
        net = ElmanRNN_v3(N,hidden_N,N)
    if args.Hregularized:
        net = ElmanRNN_pytorch_module_v2(N,hidden_N,N)
    if args.pred:
        print('Network output prediction one-step ahead', file=f)
        net = ElmanRNN_pred(N,hidden_N,N)
    if args.pred and args.Hregularized:
        print('Network output predeiction one-step ahead and Hregularized',file=f)
        net = ElmanRNN_pred_v2(N,hidden_N,N)
    if args.predfd:
        print('Predict one step ahead using feedback', file=f)
        net = ElmanRNN_pred_feedback(N,hidden_N,N)    
    if args.pred_d:
        net = ElmanRNN_pred_v3(N,hidden_N,N,args.pred_d)
        print('Network output prediction {}-step ahead'.format(str(args.pred_d)), file=f)
    if args.snn:
        if args.predfd or args.pred_d:
            raise ValueError('SNN mode currently supports base/pred/pred+Hregularized, not --predfd or --pred_d.')
        if args.snn_loss not in ('mse', 'ce'):
            raise ValueError('Unsupported --snn-loss: {} (use mse/ce)'.format(args.snn_loss))
        if args.snn_readout not in ('softmax_seq', 'logits_seq'):
            raise ValueError('Unsupported --snn-readout: {} (use softmax_seq/logits_seq)'.format(args.snn_readout))
        if args.snn_loss == 'ce' and args.snn_readout != 'logits_seq':
            raise ValueError('--snn-loss ce requires --snn-readout logits_seq.')
        if args.Hregularized and args.snn_loss == 'ce':
            raise ValueError('--Hregularized with --snn-loss ce is not supported. Use --snn-loss mse.')
        print(
            'Use SNN (event-driven) dynamics: alpha={:.3f}, beta={:.3f}, threshold={:.3f}, reset={:.3f}, refractory={}, sg_beta={:.3f}, input_mode={}, readout={}, loss={}, rate_lambda={:.2e}, learn_th={}'.format(
                args.lif_alpha,
                args.lif_beta,
                args.lif_threshold,
                args.lif_reset,
                args.lif_refractory,
                args.sg_beta,
                args.input_spike_mode,
                args.snn_readout,
                args.snn_loss,
                args.snn_rate_lambda,
                bool(args.lif_learn_threshold),
            ),
            file=f,
        )
        snn_kwargs = dict(
            lif_alpha=args.lif_alpha,
            lif_beta=args.lif_beta,
            lif_threshold=args.lif_threshold,
            lif_reset=args.lif_reset,
            sg_beta=args.sg_beta,
            input_spike_mode=args.input_spike_mode,
            input_spike_scale=args.input_spike_scale,
            lif_refractory=args.lif_refractory,
            learnable_threshold=bool(args.lif_learn_threshold),
            readout_mode=args.snn_readout,
        )
        if args.pred and args.Hregularized:
            net = ElmanSNN_pred_v2(N, hidden_N, N, **snn_kwargs)
        elif args.pred:
            net = ElmanSNN_pred(N, hidden_N, N, **snn_kwargs)
        elif args.Hregularized:
            net = ElmanSNN_v2(N, hidden_N, N, **snn_kwargs)
        else:
            net = ElmanSNN(N, hidden_N, N, **snn_kwargs)
    if args.relu and hasattr(net, 'rnn'):
        net.rnn = nn.RNN(N,hidden_N,1,batch_first=True,nonlinearity='relu')   
    if args.fixi:
        for name,p in net.named_parameters():
            if name == 'rnn.weight_ih_l0' or name == 'input_linear.weight':
                p.requires_grad = False;
                p.data.fill_(0)
                for i in range(min(p.data.shape[0], p.data.shape[1])):
                    p.data[i, i] = 1
                print('Fixing input matrix to identity matrix', file=f)
            elif name == 'rnn.bias_ih_l0' or name == 'input_linear.bias':
                p.requires_grad = False; 
                p.data.fill_(0)
                print('Fixing input bias to 0', file = f)
    if args.nobias:
        for name,p in net.named_parameters():
            if name == 'rnn.bias_hh_l0' or name == 'hidden_linear.bias':
                p.requires_grad = False;
                p.data.fill_(0)
                print('Fixing RNN bias to 0', file=f)
    if args.fixw:
        for name,p in net.named_parameters():
            if name == 'rnn.weight_hh_l0' or name == 'hidden_linear.weight':
                p.requires_grad = False;
                p.data = torch.rand(p.data.shape)*2*1/np.sqrt(N) - 1/np.sqrt(N)
                print('Fixing recurrent matrix to a random matrix', file=f)
            elif name == 'rnn.bias_hh_l0' or name == 'hidden_linear.bias':
                p.requires_grad = False; 
                p.data.fill_(0)
                print('Fixing input bias to 0', file = f )
    if args.custom and not args.snn:
        print('randomly zero out hidden unit activity', file = f)
        net = ElmanRNN_sparse(N,hidden_N,N,args.custom)
    elif args.custom and args.snn:
        print('Ignore --custom in SNN mode (only implemented for nn.RNN path)', file=f)
    if args.ac_output == 'tanh':
        net.act = nn.Tanh()
        print('Change output activation function to tanh', file = f)
    if args.ac_output == 'relu':
        net.act = nn.ReLU()
        print('Change output activation function to relu', file = f)




    # if args.ownnet:
    #     print('Written RNN instead of nn modules', file = f)
    #     net = ElmanRNN(N,hidden_N,N)

    ## MSE criteria
    criterion = nn.MSELoss(reduction='sum')
    # criterion = nn.MSELoss(reduction='mean')

    ##  load checkpoint and resume training
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading previous network '{}'".format(args.resume), file=f)
            checkpoint = torch_load_compat(args.resume)
            net.load_state_dict(checkpoint['state_dict'])
            print("=> loaded previous network '{}' ".format(args.resume), file=f)
            if args.resume_plain:
                print('Plain resume: keep current input and target unchanged', file=f)
            else:
                X = checkpoint['X_mini']; Target = checkpoint['Target_mini']
                X_new = np.copy(X);Target_new = np.copy(Target)
                idx = np.arange(np.int64(N)); np.random.seed(20); np.random.shuffle(idx)
                X_mini = X[:,:,idx]
                idx = np.arange(np.int64(N)); np.random.seed(30); np.random.shuffle(idx)
                Target_mini = Target[:,:,idx]
                print('Shuffle the original input and target', file = f)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume), file=f)

    ## if autoencoder
    if args.ae:
        print('Autoencoder scenario: Target = Input', file = f)
        Target_mini = X_mini

    print(X_mini.shape)
    # H0 value
    h0 = torch.zeros(1,X_mini.shape[0],hidden_N) # n_layers * BatchN * NHidden
    print(h0.shape)

    ## enable GPU computing (with safe fallback to CPU)
    use_cuda = bool(args.gpu) and torch.cuda.is_available()
    if args.gpu and not use_cuda:
        print('CUDA requested (gpu={}) but unavailable. Falling back to CPU.'.format(args.gpu), file=f)
    print('Cuda device availability: {}'.format(torch.cuda.is_available()), file=f)
    if use_cuda:
        criterion = criterion.cuda()
        net = net.cuda()
        X_mini = X_mini.cuda()
        Target_mini = Target_mini.cuda()
        h0 = h0.cuda()

    # construct optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    if args.adam:
        print('Using ADAM optimizer', file=f)
        optimizer = torch.optim.Adam(net.parameters(),lr=lr)

    # create weight mask or null mask
    if args.partial:
        # Currently only enable partial training for RNN weights
        print('Training sparsity:{}'.format(args.partial),file=f)
        Mask_W = np.random.uniform(0,1,(hidden_N,hidden_N))# determine the set of connections to be trained
        Mask_B = np.random.uniform(0,1,(hidden_N))
        Mask_W = Mask_W > args.partial; Mask_B = Mask_B > args.partial # True == untrained connections
        if args.nonoverlap:
            Mask_W = ~(~(Mask_W) & checkpoint['Mask_W'])
            Mask_B = ~(~(Mask_B) & checkpoint['Mask_B'])
        Mask = []
        for name,p in net.named_parameters():
            if name == 'rnn.weight_hh_l0' or name == 'hidden_linear.weight': 
                Mask.append(Mask_W); print('Partially train RNN weight',file=f)
            elif name == 'rnn.bias_hh_l0' or name == 'hidden_linear.bias': 
                Mask.append(Mask_B); print('Partially train RNN bias',file=f)
            else:
                Mask.append(np.zeros(p.shape))
    else:
        Mask = [];
        for name,p in net.named_parameters():
            Mask.append(np.zeros(p.shape))


    # For debug
    # # save network input, state 
    # save_dict = {'state_dict': net.state_dict(),
    #     'X_mini': X_mini.cpu(),
    #     'Target_mini': Target_mini.cpu(),
    #     }
    # if args.partial:
    #     save_dict['Mask'] = Mask
    #     save_dict['Mask_W'] = Mask_W; save_dict['Mask_B'] = Mask_B
    # torch.save(save_dict, args.savename+'.pth.tar')


    # start training or step-wise training
    start = time.time()
    if args.interleaved:
        net, loss_list, y_hat, hidden = train_interleaved(X_mini,Target_mini, h0, n_epochs, net, criterion, \
            optimizer, RecordEp, Mask)
    elif args.interval:
        net, loss_list, y_hat, hidden = train_interval(X_mini, Target_mini, h0, n_epochs, net, criterion, \
            optimizer, RecordEp, Mask, args.interval)
    elif args.Hregularized:
        net,loss1_list,loss2_list,y_hat,hidden = train_Hregularized(X_mini,Target_mini,h0,n_epochs,net,criterion,\
            optimizer,RecordEp,Mask,args.Hregularized/hidden_N)
        loss_list = [loss1_list, loss2_list]
        print('Add hidden unit firing cost, weight: {}'.format(args.Hregularized/hidden_N),file=f)
    elif args.pred2:
        net, loss_list, y_hat, hidden = train_everyT(X_mini,Target_mini, h0, n_epochs, net, criterion, \
            optimizer, RecordEp, Mask, args.pred2)
        print('Train at each time step and predicting {} steps into the future'.format(args.pred2),file=f)
    else:
        net, loss_list, y_hat, hidden = train_partial(X_mini,Target_mini, h0, n_epochs, net, criterion, \
            optimizer, RecordEp, Mask)
    end = time.time(); deltat= end - start;
    print('Total training time: {0:.1f} minuetes'.format(deltat/60), file=f)

    # save network input, state 
    save_dict = {'state_dict': net.state_dict(),
        'model_name': net.__class__.__name__,
        'y_hat': y_hat,
        'X_mini': X_mini.cpu(),
        'Target_mini': Target_mini.cpu(),
        'hidden': hidden,
        'loss': loss_list}
    if args.snn:
        save_dict['snn'] = 1
        save_dict['snn_standard'] = int(args.snn_standard)
        save_dict['lif_alpha'] = args.lif_alpha
        save_dict['lif_beta'] = args.lif_beta
        if hasattr(net, 'lif_threshold'):
            save_dict['lif_threshold'] = float(torch.clamp(net.lif_threshold.detach(), min=1e-4).item())
        else:
            save_dict['lif_threshold'] = args.lif_threshold
        save_dict['lif_reset'] = args.lif_reset
        save_dict['sg_beta'] = args.sg_beta
        save_dict['lif_refractory'] = args.lif_refractory
        save_dict['lif_learn_threshold'] = int(args.lif_learn_threshold)
        save_dict['input_spike_mode'] = args.input_spike_mode
        save_dict['input_spike_scale'] = args.input_spike_scale
        save_dict['snn_readout'] = args.snn_readout
        save_dict['snn_loss'] = args.snn_loss
        save_dict['snn_label_source'] = args.snn_label_source
        save_dict['snn_rate_lambda'] = args.snn_rate_lambda
        save_dict['snn_target_rate'] = args.snn_target_rate
    if args.partial:
        save_dict['Mask_W'] = Mask_W; save_dict['Mask_B'] = Mask_B
    torch.save(save_dict, args.savename+'.pth.tar')

    # plot loss function iteration curve
    plt.figure()
    if args.Hregularized:
        line1 = plt.plot(loss1_list); line2 = plt.plot(loss2_list); 
        line3 = plt.plot(np.array(loss1_list)+np.array(loss2_list));
        plt.legend(['Target loss','Hidden unit loss','Total loss']);
        plt.title('Loss iteration: lamda={:.4f}'.format(args.Hregularized/hidden_N))
    else:
        plt.plot(loss_list);plt.title('Loss iteration'); 
    plt.savefig(args.savename+'.png')


# def train(X_mini, Target_mini, h0, n_epochs, net, criterion, optimizer, RecordEp):
#     '''
#     Use SGD to train neural network
#     (y_hat and hidden only recorded for batch=1)    
#         INPUT:
#             X_mini: batchN*seqN*featureN
#             Target_mini: batchN*seqN*featureN
#             n_epochs: number of epoches to train
#             net: nn.module: pre-defined network structure
#     '''       
#     params = list(net.parameters())
#     print('{} parameters to optimize'.format(len(params)))
#     loss_list = []
#     y_hat = np.zeros((np.int(n_epochs/RecordEp),X_mini.shape[1],X_mini.shape[2]))
#     hidden = np.zeros((np.int(n_epochs/RecordEp),X_mini.shape[1],X_mini.shape[2]))
#     start = time.time()
#     for epoch in range(n_epochs):
#         output, hidden = net(X_mini, h0)
#         optimizer.zero_grad() # Clears existing gradients from previous epoch        
#         loss = criterion(output,Target_mini)
#         loss.backward() # Does backpropagation and calculates gradients
#         optimizer.step() # Updates the weights accordingly
#         loss_list = np.append(loss_list,loss.item())
#         if epoch%RecordEp == 0:
#             end = time.time(); deltat= end - start; start = time.time()
#             print('Epoch: {}/{}.............'.format(epoch,n_epochs), end=' ')
#             print("Loss: {:.4f}".format(loss.item()))
#             print('Time Elapsed since last display: {0:.1f} seconds'.format(deltat))
#             print('Estimated remaining time: {0:.1f} minutes'.format(deltat*(n_epochs-epoch)/RecordEp/60))
#             y_hat[np.int(epoch/RecordEp),:,:] = output.cpu().detach().numpy()[0,:,:]
#             hidden[np.int(epoch/RecordEp),:,:] = hidden.cpu().detach().numpy()[0,:,:]
#     return net, loss_list, y_hat, hidden

def snn_rate_regularization(net, device):
    if (not args.snn) or args.snn_rate_lambda <= 0:
        return torch.zeros((), device=device)
    spike_seq = getattr(net, 'last_spike_seq', None)
    if spike_seq is None:
        return torch.zeros((), device=device)
    spike_rate = spike_seq.mean()
    target_rate = torch.tensor(float(args.snn_target_rate), device=device, dtype=spike_rate.dtype)
    return args.snn_rate_lambda * (spike_rate - target_rate).pow(2)


def snn_last_rate(net):
    spike_seq = getattr(net, 'last_spike_seq', None)
    if spike_seq is None:
        return None
    return float(spike_seq.detach().mean().item())


def apply_grad_mask(param, mask_np):
    if param.grad is None:
        return
    mask_arr = np.asarray(mask_np)
    if mask_arr.shape == ():
        if bool(mask_arr):
            param.grad.data.zero_()
        return
    mask_t = torch.from_numpy(mask_arr.astype(bool)).to(param.grad.device)
    param.grad.data[mask_t] = 0


def extract_ce_labels(target):
    if args.snn_label_source == 'last':
        label_source = target[:, -1, :]
    elif args.snn_label_source == 'mean':
        label_source = target.mean(dim=1)
    elif args.snn_label_source == 'max':
        label_source = target.max(dim=1).values
    else:
        raise ValueError('Unsupported --snn-label-source: {} (use last/mean/max)'.format(args.snn_label_source))
    return torch.argmax(label_source, dim=-1).long()


def compute_task_loss(output, target, criterion, ignore_first=False, pred_offset=0):
    if args.snn and args.snn_loss == 'ce':
        start_idx = max(int(pred_offset), 0)
        if ignore_first:
            start_idx = max(start_idx, 1)
        if output.shape[1] <= start_idx:
            logits = output.mean(dim=1)
        else:
            logits = output[:, start_idx:, :].mean(dim=1)
        labels = extract_ce_labels(target)
        return F.cross_entropy(logits, labels)

    if pred_offset and pred_offset > 0:
        return criterion(output[:, pred_offset:, :], target[:, pred_offset:, :])
    if ignore_first:
        return criterion(output[:, 1:, :], target[:, 1:, :])
    return criterion(output, target)


def train_partial(X_mini, Target_mini, h0, n_epochs, net, criterion, optimizer, RecordEp, Mask):
    '''
    With untrainable weight mask
    Fix the intermediate recording of predloss (10/26/2021 Y.C.)
    Note first time-step info is not used (07/08/2021,Y.C.)
    Add stop criteria (07/11/2021, Y.C.)
        INPUT:
            X_mini: batchN*seqN*featureN
            Target_mini: batchN*seqN*featureN
            n_epochs: number of epoches to train
            net: nn.module: pre-defined network structure
            criterion: loss function
            optimizer: 
            RecordEp: the recording and printing frequency
            Mask: 1=untrainable parameters: a list with len(net.parameters())
        OUTPUT:
            y_hat: BatchN*RecordN*SeqN*HN
            hidden: BatchN*RecordN*SeqN*HN
    '''       
    params = list(net.parameters()); print('{} parameters to optimize'.format(len(params)))
    loss_list = []
    h_t = h0; batch_size,SeqN,N = X_mini.shape; _,_,hidden_N = h_t.shape
    y_hat = np.zeros((batch_size,np.int64(n_epochs/RecordEp),SeqN,N))
    hidden = np.zeros((batch_size,np.int64(n_epochs/RecordEp),SeqN,hidden_N))
    start = time.time(); epoch = 0; stop = 0;
    while stop == 0 and epoch < n_epochs:
        if args.lr_step:
            lr_step = list(map(int, args.lr_step.split(',')))
            if epoch in lr_step:
                print('Decrease lr to 50per at epoch {}'.format(epoch), file = f)
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
        if args.continuous:
            output, h_t = net(X_mini,h_t.detach())
        else: 
            output, h_t = net(X_mini,h0)
        optimizer.zero_grad() # Clears existing gradients from previous epoch
        loss_task = compute_task_loss(
            output,
            Target_mini,
            criterion,
            ignore_first=(args.pred_d == 0),
            pred_offset=args.pred_d,
        )
        loss_sparse = snn_rate_regularization(net, X_mini.device)
        loss = loss_task + loss_sparse
        if not torch.isfinite(loss):
            print('Non-finite loss at epoch {}. Stop training early.'.format(epoch), file=f)
            break
        loss.backward() # Does backpropagation and calculates gradients
        for l,p in enumerate(net.parameters()):
            if p.requires_grad and p.grad is not None:
                apply_grad_mask(p, Mask[l])
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
        optimizer.step() # Updates the weights accordingly
        if epoch > 1000:
            diff = [loss_list[i+1]-loss_list[i] for i in range(len(loss_list)-1)]
            mean_diff = np.mean(abs(np.array(diff[-5:-1])))
            # init_loss = np.mean(np.array(loss_list[0:10]))
            init_loss = np.mean(np.array(loss_list[0]))
            if mean_diff < loss.item()*0.00001 and loss.item() < init_loss*0.010: 
                stop = 1
        loss_list = np.append(loss_list,loss.item())
        if epoch%RecordEp == 0:
            end = time.time(); deltat= end - start; start = time.time()
            print('Epoch: {}/{}.............'.format(epoch,n_epochs), end=' ')
            print("Loss: {:.4f}".format(loss.item()))
            if args.snn:
                spike_rate = snn_last_rate(net)
                if spike_rate is not None:
                    print('Spike rate: {:.5f}, sparse loss: {:.5f}'.format(
                        spike_rate, float(loss_sparse.detach().item())
                    ))
            print('Time Elapsed since last display: {0:.1f} seconds'.format(deltat))
            print('Estimated remaining time: {0:.1f} minutes'.format(deltat*(n_epochs-epoch)/RecordEp/60))
            if args.pred:
                hidden_seq = np.zeros((batch_size,SeqN,hidden_N))
                output = output.cpu().detach().numpy()
            elif args.continuous:
                output, hidden_seq = evaluate_onestep(X_mini,Target_mini, h_t, net, criterion)
            else:
                output, hidden_seq = evaluate_onestep(X_mini,Target_mini, h0, net, criterion)
            y_hat[:,np.int64(epoch/RecordEp),:,:] = output
            hidden[:,np.int64(epoch/RecordEp),:,:] = hidden_seq
            if args.noisytrain:
                sd1 = args.noisy*X_mini.cpu().numpy().max()/100.
                sd2 = args.noisy*Target_mini.cpu().numpy().max()/100.
                X_mini = X_mini + torch.tensor(np.random.normal(0,1,X_mini.shape).astype(np.single)*sd1).to(X_mini.device)
                Target_mini = Target_mini + torch.tensor(np.random.normal(0,1,Target_mini.shape).astype(np.single)*sd2).to(Target_mini.device)
                X_mini.detach(); Target_mini.detach()         
        epoch = epoch + 1
    return net, loss_list, y_hat, hidden


def train_everyT(X_mini, Target_mini, h0, n_epochs, net, criterion, optimizer, RecordEp, Mask, k=1):
    '''
    Modified from train_partial.py (07/17/2021)
    Collect loss function from every time step
    Predict k steps into the future
        INPUT:
            X_mini: batchN*seqN*featureN
            Target_mini: batchN*seqN*featureN
            n_epochs: number of epoches to train
            net: nn.module: pre-defined network structure
            criterion: loss function
            optimizer: 
            RecordEp: the recording and printing frequency
            Mask: 1=untrainable parameters: a list with len(net.parameters())
            k: number of steps predicting into the future
        OUTPUT:
            y_hat: BatchN*RecordN*SeqN*HN
            hidden: BatchN*RecordN*SeqN*HN
    '''       
    params = list(net.parameters()); print('{} parameters to optimize'.format(len(params)))
    loss_list = []
    h_t = h0; batch_size,SeqN,N = X_mini.shape; _,_,hidden_N = h_t.shape
    y_hat = np.zeros((batch_size,np.int64(n_epochs/RecordEp),SeqN,N))
    hidden = np.zeros((batch_size,np.int64(n_epochs/RecordEp),SeqN,hidden_N))
    start = time.time(); epoch = 0; stop = 0;
    while stop == 0 and epoch < n_epochs:
        # adaptively adjust learning rate
        if args.lr_step:
            lr_step = list(map(int, args.lr_step.split(',')))
            if epoch in lr_step:
                print('Decrease lr to 50per at epoch {}'.format(epoch), file = f)
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
        # form an average loss function
        loss_list_pre = []
        for t in np.arange(SeqN-k):
            X_t = X_mini[:,:t+k,:]
            X_t[:,t+1:t+k,:] = 0
            Target_t = Target_mini[:,:t+k,:]
            output, h_t = net(X_t,h0)
            optimizer.zero_grad() # Clears existing gradients from previous epoch        
            loss_task = compute_task_loss(output, Target_t, criterion, ignore_first=False, pred_offset=0)
            loss_sparse = snn_rate_regularization(net, X_mini.device)
            loss = loss_task + loss_sparse
            loss.backward() # Does backpropagation and calculates gradients
            for l,p in enumerate(net.parameters()):
                if p.requires_grad and p.grad is not None:
                    apply_grad_mask(p, Mask[l])
            optimizer.step() # Updates the weights accordingly
            loss_list_pre = np.append(loss_list_pre,loss.item())
        loss_list = np.append(loss_list,np.mean(loss_list_pre))
        if epoch > 1000:
            diff = [loss_list[i+1]-loss_list[i] for i in range(len(loss_list)-1)]
            mean_diff = np.mean(abs(np.array(diff[-5:-1])))
            init_loss = np.mean(np.array(loss_list[0:10]))
            if mean_diff < loss.item()*0.01 and loss.item() < init_loss*0.1: 
                stop = 1
        if epoch%RecordEp == 0:
            end = time.time(); deltat= end - start; start = time.time()
            print('Epoch: {}/{}.............'.format(epoch,n_epochs), end=' ')
            print("Loss: {:.4f}".format(loss.item()))
            if args.snn:
                spike_rate = snn_last_rate(net)
                if spike_rate is not None:
                    print('Spike rate: {:.5f}, sparse loss: {:.5f}'.format(
                        spike_rate, float(loss_sparse.detach().item())
                    ))
            print('Time Elapsed since last display: {0:.1f} seconds'.format(deltat))
            print('Estimated remaining time: {0:.1f} minutes'.format(deltat*(n_epochs-epoch)/RecordEp/60))
            if args.continuous:
                output, hidden_seq = evaluate_onestep(X_mini,Target_mini, h_t, net, criterion)
            else:
                output, hidden_seq = evaluate_onestep(X_mini,Target_mini, h0, net, criterion)
            y_hat[:,np.int64(epoch/RecordEp),:,:] = output
            hidden[:,np.int64(epoch/RecordEp),:,:] = hidden_seq
            if args.noisytrain:
                sd1 = args.noisy*X_mini.cpu().numpy().max()/100.
                sd2 = args.noisy*Target_mini.cpu().numpy().max()/100.
                X_mini = X_mini + torch.tensor(np.random.normal(0,1,X_mini.shape).astype(np.single)*sd1).to(X_mini.device)
                Target_mini = Target_mini + torch.tensor(np.random.normal(0,1,Target_mini.shape).astype(np.single)*sd2).to(Target_mini.device)
                X_mini.detach(); Target_mini.detach()         
        epoch = epoch + 1
    return net, loss_list, y_hat, hidden



def train_interleaved(X_mini, Target_mini, h0, n_epochs, net, criterion, optimizer, RecordEp, Mask):
    '''
    Perform interleaved training: train one sample, update weight, train another sample, update again
        INPUT:
            X_mini: batchN*seqN*featureN
            Target_mini: batchN*seqN*featureN
            n_epochs: number of epoches to train
            net: nn.module: pre-defined network structure
            criterion: loss function
            optimizer: 
            RecordEp: the recording and printing frequency
            Mask: 1=untrainable parameters: a list with len(net.parameters())
        OUTPUT:
            loss: length: batch_size*epoch
    '''       
    params = list(net.parameters()); print('{} parameters to optimize'.format(len(params)))
    batch_size = X_mini.shape[0]
    loss_list = []
    h_t = h0
    y_hat = np.zeros((batch_size,np.int64(n_epochs/RecordEp),X_mini.shape[1],X_mini.shape[2]))
    hidden = np.zeros((batch_size,np.int64(n_epochs/RecordEp),X_mini.shape[1],h_t.shape[2]))
    start = time.time()
    for epoch in range(n_epochs):
        for b in np.arange(batch_size):
            if args.continuous:
                output, h_t = net(X_mini[b:b+1,:,:],h_t[:,b:b+1,:].detach())
            else: 
                output, h_t = net(X_mini[b:b+1,:,:],h0[:,b:b+1,:])
            optimizer.zero_grad() # Clears existing gradients from previous epoch        
            loss_task = compute_task_loss(output, Target_mini[b:b+1,:,:], criterion, ignore_first=False, pred_offset=0)
            loss_sparse = snn_rate_regularization(net, X_mini.device)
            loss = loss_task + loss_sparse
            loss.backward() # Does backpropagation and calculates gradients
            for l,p in enumerate(net.parameters()):
                if p.grad is not None:
                    apply_grad_mask(p, Mask[l])
            optimizer.step() # Updates the weights accordingly
            loss_list = np.append(loss_list,loss.item())
            if epoch%RecordEp == 0:
                end = time.time(); deltat= end - start; start = time.time()
                print('Epoch: {}/{}.............'.format(epoch,n_epochs), end=' ')
                print("Loss: {:.4f}".format(loss.item()))
                if args.snn:
                    spike_rate = snn_last_rate(net)
                    if spike_rate is not None:
                        print('Spike rate: {:.5f}, sparse loss: {:.5f}'.format(
                            spike_rate, float(loss_sparse.detach().item())
                        ))
                print('Time Elapsed since last display: {0:.1f} seconds'.format(deltat))
                print('Estimated remaining time: {0:.1f} minutes'.format(deltat*(n_epochs-epoch)/RecordEp/60))
                if args.continuous:
                    output, hidden_seq = evaluate_onestep(X_mini[b:b+1,:,:],Target_mini[b:b+1,:,:], h_t[:,b:b+1,:], net, criterion)
                else:
                    output, hidden_seq = evaluate_onestep(X_mini[b:b+1,:,:],Target_mini[b:b+1,:,:], h0[:,b:b+1,:], net, criterion)
                y_hat[b,np.int64(epoch/RecordEp),:,:] = output[0,:,:]
                hidden[b,np.int64(epoch/RecordEp),:,:] = hidden_seq[0,:,:]
    return net, loss_list, y_hat, hidden


def train_interval(X_mini, Target_mini, h0, n_epochs, net, criterion, optimizer, RecordEp, Mask, interval):
    '''
    Perform interleaved training with user-defined interval: train one sample for multiple times (interval)\
    ,update weight, train another sample for multiple times, update again
        INPUT:
            X_mini: batchN*seqN*featureN
            Target_mini: batchN*seqN*featureN
            n_epochs: number of epoches to train
            net: nn.module: pre-defined network structure
            criterion: loss function
            optimizer: 
            RecordEp: the recording and printing frequency
            Mask: 1=untrainable parameters: a list with len(net.parameters())
            interval: training interval between two training samples
        OUTPUT:
            y_hat, hidden: only record at the last interval
            loss: length: batch_size*epoch
    '''       
    params = list(net.parameters()); print('{} parameters to optimize'.format(len(params)))
    batch_size = X_mini.shape[0]
    loss_list = []
    h_t = h0
    y_hat = np.zeros((batch_size,np.int64(n_epochs/RecordEp),X_mini.shape[1],X_mini.shape[2]))
    hidden = np.zeros((batch_size,np.int64(n_epochs/RecordEp),X_mini.shape[1],h_t.shape[2]))
    start = time.time()
    for epoch in range(n_epochs):
        for b in np.arange(batch_size):
            for it in np.arange(interval):
                if args.continuous:
                    output, h_t = net(X_mini[b:b+1,:,:],h_t[:,b:b+1,:].detach())
                else: 
                    output, h_t = net(X_mini[b:b+1,:,:],h0[:,b:b+1,:])
                optimizer.zero_grad() # Clears existing gradients from previous epoch        
                loss_task = compute_task_loss(output, Target_mini[b:b+1,:,:], criterion, ignore_first=False, pred_offset=0)
                loss_sparse = snn_rate_regularization(net, X_mini.device)
                loss = loss_task + loss_sparse
                loss.backward() # Does backpropagation and calculates gradients
                for l,p in enumerate(net.parameters()):
                    if p.requires_grad and p.grad is not None:
                        apply_grad_mask(p, Mask[l])
                optimizer.step() # Updates the weights accordingly
                loss_list = np.append(loss_list,loss.item())
            if epoch%RecordEp == 0:
                end = time.time(); deltat= end - start; start = time.time()
                print('Epoch: {}/{}.............'.format(epoch,n_epochs), end=' ')
                print("Loss: {:.4f}".format(loss.item()))
                if args.snn:
                    spike_rate = snn_last_rate(net)
                    if spike_rate is not None:
                        print('Spike rate: {:.5f}, sparse loss: {:.5f}'.format(
                            spike_rate, float(loss_sparse.detach().item())
                        ))
                print('Time Elapsed since last display: {0:.1f} seconds'.format(deltat))
                print('Estimated remaining time: {0:.1f} minutes'.format(deltat*(n_epochs-epoch)/RecordEp/60))
                if args.continuous:
                    output, hidden_seq = evaluate_onestep(X_mini[b:b+1,:,:],Target_mini[b:b+1,:,:], h_t[:,b:b+1,:], net, criterion)
                else:
                    output, hidden_seq = evaluate_onestep(X_mini[b:b+1,:,:],Target_mini[b:b+1,:,:], h0[:,b:b+1,:], net, criterion)
                y_hat[b,np.int64(epoch/RecordEp),:,:] = output[0,:,:]
                hidden[b,np.int64(epoch/RecordEp),:,:] = hidden_seq[0,:,:]
    return net, loss_list, y_hat, hidden




def train_Hregularized(X_mini, Target_mini, h0, n_epochs, net, criterion, optimizer, RecordEp, Mask, lamda):
    '''
    Training with l2 regularization on hidden unit activity
    Note: use Elman_pytorch_module_v2 as net (output hidden unit: BatchN*SeqN*HiddenN)
    Note: the second output of net need to contain time sequence of hidden unit activity
    '''       
    params = list(net.parameters()); print('{} parameters to optimize'.format(len(params)))
    loss1_list = []; loss2_list = []; loss_list = []
    h_t = h0; batch_size,SeqN,N = X_mini.shape; hidden_N = h_t.shape[2]
    y_hat = np.zeros((batch_size,np.int64(n_epochs/RecordEp),SeqN,N))
    hidden = np.zeros((batch_size,np.int64(n_epochs/RecordEp),SeqN,hidden_N))
    start = time.time(); epoch = 0; stop = 0
    while stop == 0 and epoch < n_epochs:
        if args.lr_step:
            lr_step = list(map(int, args.lr_step.split(',')))
            if epoch in lr_step:
                print('Decrease lr to 50per at epoch {}'.format(epoch), file = f)
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
        if args.continuous:
            output, h_t = net(X_mini,h_t[:,-1:,:].detach())
        else: 
            output, h_t = net(X_mini,h0)
        optimizer.zero_grad() # Clears existing gradients from previous epoch        
        loss1 = compute_task_loss(output, Target_mini, criterion, ignore_first=True, pred_offset=0)
        loss2 = lamda*criterion(h_t,torch.zeros(h_t.shape).to(X_mini.device))
        loss_sparse = snn_rate_regularization(net, X_mini.device)
        loss1_list.append(loss1.item()); loss2_list.append(loss2.item())
        loss = loss1 + loss2 + loss_sparse; loss_list.append(loss.item())
        loss.backward() # Does backpropagation and calculates gradients
        for l,p in enumerate(net.parameters()):
            if p.requires_grad and p.grad is not None:
                apply_grad_mask(p, Mask[l])
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
        optimizer.step() # Updates the weights accordingly
        if epoch > 1000:
            diff = [loss_list[i+1]-loss_list[i] for i in range(len(loss_list)-1)]
            mean_diff = np.mean(abs(np.array(diff[-5:-1])))
            init_loss = np.mean(np.array(loss_list[0]))
            if mean_diff < loss.item()*0.01 and loss.item() < init_loss*0.1: 
                stop = 1
        if epoch%RecordEp == 0:
            end = time.time(); deltat= end - start; start = time.time()
            print('Epoch: {}/{}.............'.format(epoch,n_epochs), end=' ')
            print("Loss: {:.4f}".format(loss.item()))
            if args.snn:
                spike_rate = snn_last_rate(net)
                if spike_rate is not None:
                    print('Spike rate: {:.5f}, sparse loss: {:.5f}'.format(
                        spike_rate, float(loss_sparse.detach().item())
                    ))
            print('Time Elapsed since last display: {0:.1f} seconds'.format(deltat))
            print('Estimated remaining time: {0:.1f} minutes'.format(deltat*(n_epochs-epoch)/RecordEp/60))
            y_hat[:,np.int64(epoch/RecordEp),:,:] = output.cpu().detach().numpy()
            hidden[:,np.int64(epoch/RecordEp),:,:] = h_t.cpu().detach().numpy()
        epoch = epoch + 1
    return net, loss1_list, loss2_list, y_hat, hidden









# def train_continuH(X_mini, Target_mini, h0, n_epochs, net, criterion, optimizer, RecordEp, Mask):
#     '''
#     Loop the entire sequence each epoch
#     (y_hat and hidden only recorded for batch=1)
#         INPUT:
#             X_mini: batchN*seqN*featureN
#             Target_mini: batchN*seqN*featureN
#             n_epochs: number of epoches to train
#             net: nn.module: pre-defined network structure
#     '''       
#     params = list(net.parameters())
#     print('{} parameters to optimize'.format(len(params)))
#     loss_list = []
#     y_hat = np.zeros((np.int(n_epochs/RecordEp),X_mini.shape[1],X_mini.shape[2]))
#     hidden = np.zeros((np.int(n_epochs/RecordEp),X_mini.shape[1],X_mini.shape[2]))
#     h_t = h0
#     start = time.time()
#     for epoch in range(n_epochs):
#         h_seq = np.zeros(X_mini.shape); output_seq = np.zeros(X_mini.shape)
#         optimizer.zero_grad() # Clears existing gradients from previous epoch    
#         loss = 0; h_t = h_t.detach() # trucated BPTT, only BP one epoch                     
#         for t in np.arange(X_mini.shape[1]):
#             o_t,h_t = net(X_mini[:,t:t+1,:],h_t)
#             output_seq[:,t,:] = o_t.cpu().detach()
#             h_seq[:,t,:] = h_t.cpu().detach()
#             loss += criterion(o_t,Target_mini[:,t:t+1,:]); 
#         loss.backward()
#         for l,p in enumerate(net.parameters()):
#             p.grad.data[Mask[l]] = 0
#         optimizer.step()    
#         loss_list = np.append(loss_list,loss.item())
#         if epoch%RecordEp == 0:
#             end = time.time(); deltat= end - start; start = time.time()
#             print('Epoch: {}/{}.............'.format(epoch,n_epochs), end=' ')
#             print("Loss: {:.4f}".format(loss.item()))
#             print('Time Elapsed since last display: {0:.1f} seconds'.format(deltat))
#             print('Estimated remaining time: {0:.1f} minutes'.format(deltat*(n_epochs-epoch)/RecordEp/60))
#             y_hat[np.int(epoch/RecordEp),:,:] = output_seq[0,:,:] 
#             hidden[np.int(epoch/RecordEp),:,:] = h_seq[0,:,:]
#     return net, loss_list, y_hat, hidden



def BellShape_input(N,TotalSteps):
    # generate bellshape circular input for N*TotalSteps
    X = np.zeros((np.int64(N),np.int64(TotalSteps))) # input N*T
    Target = np.copy(X) # pre-defined target firing rate
    tmp = np.linspace(norm.ppf(0.05),norm.ppf(0.95), np.int64(TotalSteps/2))
    BellShape = norm.pdf(tmp) # Bellshape vector
    template = np.concatenate((BellShape,np.zeros(np.int64(TotalSteps/2))))
    X = np.zeros((np.int64(N),np.int64(TotalSteps)))# time-shifting matrix
    for i in np.arange(np.int64(N)):
        X[i,:] = np.roll(template,np.int64(i*(TotalSteps/N)))
    X  = X / np.sum(X,0) # Normalize X and Target by column
    idx = np.arange(np.int64(N)); np.random.seed(10); np.random.shuffle(idx)
    Target = X[idx,:]
    return X, Target


def Cos_input(N,TotalSteps,T=2):
    '''
    Generate cos-shape input with phase offsets for N*TotalSteps
    INPUT:
        T: number of oscillatory periods
    '''
    X = np.zeros((np.int64(N),np.int64(TotalSteps))) # input N*T
    Target = np.copy(X) # pre-defined target firing rate
    omega = 2*np.pi/(TotalSteps/T)
    phi = TotalSteps/T/N
    for i in np.arange(np.int64(N)):
        X[i,:] = np.cos(omega*np.arange(TotalSteps)+i*phi)
    idx = np.arange(np.int64(N)); np.random.seed(10); np.random.shuffle(idx)
    Target = X[idx,:]
    return X, Target


def evaluate_onestep(X_mini, Target_mini, h_t, net, criterion):
    '''
    Loop over entire sequence to record hidden activity
    '''
    batch_size,SeqN,N = X_mini.shape
    _,_,hidden_N = h_t.shape
    if args.snn:
        h_seed = h_t.detach()
        output, _ = net(X_mini, h_seed)
        output_seq = output.cpu().detach().numpy()
        mem_seq = getattr(net, 'last_membrane_seq', None)
        if mem_seq is None:
            h_seq = np.zeros((batch_size, SeqN, hidden_N))
        else:
            h_seq = mem_seq.cpu().detach().numpy()
        return output_seq, h_seq

    h_seq = np.zeros((batch_size,SeqN,hidden_N)); output_seq = np.zeros(X_mini.shape)
    h_t = h_t.detach()
    for t in np.arange(X_mini.shape[1]):
        o_t,h_t = net(X_mini[:,t:t+1,:],h_t)
        output_seq[:,t:t+1,:] = o_t.cpu().detach().numpy()
        h_seq[:,t:t+1,:] = h_t.cpu().detach().numpy().transpose((1,0,2))
    return output_seq, h_seq





if __name__ == '__main__':
    main()
