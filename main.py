import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import lunzi as lz
import scipy.io
import torch.nn.functional as F
from lunzi.typing import *
from opt import GroupRMSprop
from rotmap import * 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class FLAGS(lz.BaseFLAGS):
    problem = ''
    gt_path = ''
    obs_path = ''
    dataset = ''
    depth = 1
    n_iters = 1000000
    n_dev_iters = max(1, n_iters // 1000)
    init_scale = 0.001  # average magnitude of entries
    shape = [0, 0]
    optimizer = 'GroupRMSprop'
    initialization = 'gaussian'  # `orthogonal` or `identity` or `gaussian`
    lr = 0.01
    train_thres = 1.e-5
    loss_fn="l1"
    unobs_path= ''
    gt_mat=''
    add_reg=0
    is_synthetic=False
    hidden_sizes = []

    @classmethod
    def finalize(cls):
        assert cls.problem
        cls.add('hidden_sizes', [cls.shape[0]] + [cls.shape[1]] * cls.depth, overwrite_false=True)



def get_e2e(model):
    weight = None
    for fc in model.children():
        assert isinstance(fc, nn.Linear) and fc.bias is None
        if weight is None:
            weight = fc.weight.t()
        else:
            weight = fc(weight)
    return weight

@FLAGS.inject
def init_model(model, *, hidden_sizes, initialization, init_scale, _log):
    depth = len(hidden_sizes) - 1

    if initialization == 'orthogonal':
        scale = (init_scale * np.sqrt(hidden_sizes[0]))**(1. / depth)
        matrices = []
        for param in model.parameters():
            nn.init.orthogonal_(param)
            param.data.mul_(scale)
            matrices.append(param.data.cpu().numpy())
        
    elif initialization == 'identity':
        scale = init_scale**(1. / depth)
        for param in model.parameters():
            nn.init.eye_(param)
            param.data.mul_(scale)
    elif initialization == 'gaussian':
        n = hidden_sizes[0]
        assert hidden_sizes[0] == hidden_sizes[-1]
        scale = init_scale**(1. / depth) * n**(-0.5)
        for param in model.parameters():
            nn.init.normal_(param, std=scale)
        e2e = get_e2e(model).detach().cpu().numpy()
        e2e_fro = np.linalg.norm(e2e, 'fro')
        desired_fro = FLAGS.init_scale * np.sqrt(n)
        _log.info(f"[check] e2e fro norm: {e2e_fro:.6e}, desired = {desired_fro:.6e}")
       
    elif initialization == 'uniform':
        n = hidden_sizes[0]
        # assert hidden_sizes[0] == hidden_sizes[-1]
        scale = np.sqrt(3.) * init_scale**(1. / depth) * n**(-0.5)
        for param in model.parameters():
            nn.init.uniform_(param, a=-scale, b=scale)
        e2e = get_e2e(model).detach().cpu().numpy()
        e2e_fro = np.linalg.norm(e2e, 'fro')
        desired_fro = FLAGS.init_scale * np.sqrt(n)
        _log.info(f"[check] e2e fro norm: {e2e_fro:.6e}, desired = {desired_fro:.6e}")
        
    else:
        assert 0


class BaseProblem:
    def get_d_e2e(self, e2e):
        pass

    def get_train_loss(self, e2e):
        pass

    def get_test_loss(self, e2e):
        pass


class MatrixCompletion(BaseProblem):
    ys: torch.Tensor

    @FLAGS.inject
    def __init__(self, *, gt_path, obs_path, unobs_path, gt_mat):
        self.w_gt = torch.load(gt_path, map_location=device)
        (self.us, self.vs), self.ys_ = torch.load(obs_path, map_location=device)
        (self.us_un , self.vs_un) , self.ys_un = torch.load(unobs_path ,map_location=device )
        self.unfold = torch.nn.Unfold(kernel_size = 3, stride = 3)
        self.ground_truth = torch.from_numpy(scipy.io.loadmat(gt_mat)['R_gt_c'].transpose(2,0,1))
        self.ncams = scipy.io.loadmat(gt_mat)['ncams_c'][0][0]
        
    def get_train_loss(self, e2e,  criterion=None):
        self.ys = e2e[self.us, self.vs]
        loss = criterion(self.ys.to(device).float(), self.ys_.float())
      
        if FLAGS.add_reg>0:
            out = self.unfold(e2e.unsqueeze(0).unsqueeze(0).float())[0].T.reshape([-1,3,3])
            reg = torch.norm(torch.matmul(out , out.permute(1,2,0).T) - torch.eye(3).to(device),p=2 , dim = (1,2)).mean()
            loss = loss + FLAGS.add_reg*reg
        return loss

    def get_test_loss(self, e2e ,criterion=None):

        return criterion(self.w_gt , e2e.to(device)) 

    def unobserved_loss(self , e2e , criterion):

        pred =e2e.detach().clone()
        val = pred[self.us_un, self.vs_un]
        loss = criterion(val.to(device).float() , self.ys_un.float())

        return loss
        
    
    def get_eval_loss(self, e2e, method = "median"):
        temp = e2e.detach().clone().cpu().numpy()
        for i in range(0, temp.shape[0], 3):
            temp[i:i+3 , i:i+3] = np.eye(3)
        R = torch.from_numpy(convert_mat(temp, self.ncams).transpose(2,0,1))
        gt = self.ground_truth.detach().clone()
        E_mean , E_median , E_var = compare_rot_graph(R, gt, method = method)
        return (E_mean , E_median , E_median)


        
    @FLAGS.inject
    def get_d_e2e(self, e2e, shape):
        d_e2e = torch.zeros(shape, device=device).type(torch.float64)
        d_e2e[self.us, self.vs] = self.ys - self.ys_
        d_e2e = d_e2e / len(self.ys_)
        return d_e2e


@lz.main(FLAGS)
@FLAGS.inject
def main(*, depth, hidden_sizes, n_iters, problem, train_thres, _seed, _log, _writer, _info, _fs):
    prob: BaseProblem
    if problem == 'matrix-completion':
        prob = MatrixCompletion()
    else:
        raise ValueError

    layers = zip(hidden_sizes, hidden_sizes[1:])
    model = nn.Sequential(*[nn.Linear(f_in, f_out, bias=False) for (f_in, f_out) in layers]).to(device)
    _log.info(model)

    if FLAGS.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), FLAGS.lr, momentum = 0.9  )
    elif FLAGS.optimizer == 'GroupRMSprop':
        optimizer = GroupRMSprop(model.parameters(), FLAGS.lr, eps=1e-4)
    elif FLAGS.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), FLAGS.lr)
    else:
        raise ValueError


    init_model(model)
    loss = None
    

    best_E_mean  = 999999
    best_E_var = 0
    best_E_med = 0 

    if FLAGS.loss_fn == "l1":
        criterion = torch.nn.L1Loss()
    else:
        criterion = None
    
    method = "median"
    for T in range(n_iters):
        e2e = get_e2e(model)

        loss = prob.get_train_loss(e2e , criterion = criterion)

        optimizer.zero_grad()
        loss.backward()

        with torch.no_grad():
            test_loss = prob.get_test_loss(e2e , criterion = criterion)
            unobs_loss = prob.unobserved_loss(e2e , criterion)
           

            if T % FLAGS.n_dev_iters == 0 or loss.item() <= train_thres:

                if isinstance(optimizer, GroupRMSprop):
                    adjusted_lr = optimizer.param_groups[0]['adjusted_lr']
                else:
                    adjusted_lr = optimizer.param_groups[0]['lr']

                E_mean , E_median , E_var = prob.get_eval_loss(e2e, method=method)

                _log.info(f"Iter #{T}: train = {loss.item():.3e}, test = {test_loss.item():.3e}, Mean = {E_mean}, Median = {E_median}, Var = {E_var}, lr = {adjusted_lr}")

                _writer.add_scalar('loss/train', loss.item(), global_step=T)
                _writer.add_scalar('loss/test', test_loss, global_step=T)
                _writer.add_scalar('loss/E_Mean', E_mean, global_step=T)
                _writer.add_scalar('loss/E_Median', E_median, global_step=T)
                _writer.add_scalar('loss/E_Var', E_var, global_step=T)
                _writer.add_scalar('loss/Unobs_Loss', unobs_loss, global_step=T)

                if(E_mean<best_E_mean):
                    torch.save(e2e, _fs.resolve("$LOGDIR/best.npy"))
                  
                    best_E_mean = E_mean
                    best_E_med = E_median
                    best_E_var = E_var

                if loss.item() <= train_thres:
                    break
        optimizer.step()

    _log.info(f"train loss = {loss.item()}. test loss = {test_loss.item()}")
    _log.info(f"E_mean = {best_E_mean}, E_median = {best_E_med} , E_var = {best_E_var}")
    torch.save(e2e, _fs.resolve("$LOGDIR/final.npy"))



if __name__ == '__main__':
    main()
