import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cvxpy as cvx
import os
import wandb
import lunzi as lz
import scipy.io
import torch.nn.functional as F
import json
from lunzi.typing import *
from opt import GroupRMSprop
from rotmap import * 
# import robust_loss_pytorch.general

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class FLAGS(lz.BaseFLAGS):
    problem = ''
    gt_path = ''
    obs_path = ''
    dataset = ''
    depth = 1
    n_train_samples = 0
    n_iters = 1000000
    n_dev_iters = max(1, n_iters // 1000)
    init_scale = 0.001  # average magnitude of entries
    shape = [0, 0]
    n_singulars_save = 0
    delta = 1
    optimizer = 'GroupRMSprop'
    initialization = 'gaussian'  # `orthogonal` or `identity` or `gaussian`
    lr = 0.01
    train_thres = 1.e-5
    loss_fn="l1"
    # act = "tanh"
    unobs_path= ''
    incomp_path=''
    fraction_missing=0
    outlier_probabilty=0
    cameras=0
    project_name=''
    add_reg=0
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
            # weight = act(weight)
        else:
            weight = (fc(weight))            
    return weight

def get_e2e_rand(model, input):
    model.double()
    weight = input.double()
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
        # for a, b in zip(matrices, matrices[1:]):
        #     assert np.allclose(a.dot(a.T), b.T.dot(b), atol=1e-6)
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
        # assert 0.8 <= e2e_fro / desired_fro <= 1.2
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
        # assert 0.8 <= e2e_fro / desired_fro <= 1.2
    else:
        assert 0


class BaseProblem:
    def get_d_e2e(self, e2e):
        pass

    def get_train_loss(self, e2e):
        pass

    def get_test_loss(self, e2e):
        pass

    def get_cvx_opt_constraints(self, x) -> list:
        pass


@FLAGS.inject
def cvx_opt(prob: BaseProblem, *, shape, _log: Logger, _writer: SummaryWriter, _fs: FileStorage):
    x = cvx.Variable(shape=shape)

    objective = cvx.Minimize(cvx.norm(x, 'nuc'))
    constraints = prob.get_cvx_opt_constraints(x)

    problem = cvx.Problem(objective, constraints)
    problem.solve(solver=cvx.SCS, verbose=True, use_indirect=False)
    e2e = torch.from_numpy(x.value).float()

    train_loss = prob.get_train_loss(e2e)
    test_loss = prob.get_test_loss(e2e)

    nuc_norm = e2e.norm('nuc')
    _log.info(f"train loss = {train_loss.item():.3e}, "
              f"test error = {test_loss.item():.3e}, "
              f"nuc_norm = {nuc_norm.item():.3f}")
    _writer.add_scalar('loss/train', train_loss.item())
    _writer.add_scalar('loss/test', test_loss.item())
    _writer.add_scalar('nuc_norm', nuc_norm.item())

    torch.save(e2e, _fs.resolve('$LOGDIR/nuclear.npy'))


class MatrixCompletion(BaseProblem):
    ys: torch.Tensor

    @FLAGS.inject
    def __init__(self, *, gt_path, obs_path , unobs_path, incomp_path):
        self.w_gt = torch.load(gt_path, map_location=device)
        (self.us, self.vs), self.ys_ = torch.load(obs_path, map_location=device)
        (self.us_un , self.vs_un) , self.ys_un = torch.load(unobs_path ,map_location=device )
        # self.incomp_mat = torch.load(incomp_path, map_location=device)
        self.unfold = torch.nn.Unfold(kernel_size = 3, stride = 3)
        
    def get_train_loss(self, e2e,alpha = None , scale = None ,  criterion=None):
        self.ys = e2e[self.us, self.vs]
        residual = (self.ys - self.ys_).type(torch.float32)
        if FLAGS.loss_fn=="l1":
            loss = criterion(self.ys.to(device).float(), self.ys_.float())
        else:
            loss = (self.ys.to(device) - self.ys_).pow(2).mean() # L2

        if FLAGS.add_reg>0:
            out = self.unfold(e2e.unsqueeze(0).unsqueeze(0).float())[0].T.reshape([-1,3,3])
            reg = torch.norm(torch.matmul(out , out.permute(1,2,0).T) - torch.eye(3).to(device),p=2 , dim = (1,2)).mean()

            loss = loss + FLAGS.add_reg*reg
        return (loss , residual)

    def get_test_loss(self, e2e,alpha = None  , scale = None ,criterion=None):
        residual = (e2e.to(device)- self.w_gt ).reshape(-1)
        if FLAGS.loss_fn == "l1":
            loss = criterion(self.w_gt , e2e.to(device)) # Huber / L1
        else:
            loss = (self.w_gt - e2e.to(device)).reshape(-1).pow(2).mean()

        return loss , residual 

    def unobserved_loss(self , e2e , criterion):
        pred =e2e.detach().clone()
        val = pred[self.us_un, self.vs_un]
        if FLAGS.loss_fn == "l1":
            loss = criterion(val.to(device).float() , self.ys_un.float())
        else:
            loss = (val.to(device).float() - self.ys_un.float()).pow(2).mean()  # L2
        #loss = criterion(val.to(device).float() , self.ys_un.float())
        # loss = (val.to(device).float() - self.ys_un.float()).pow(2).mean()  # L2
        return loss
        
    
    def get_eval_loss(self , e2e , ground_truth, ncams, method = "median"):
        temp = e2e.detach().clone().cpu().numpy()
        for i in range(0 , temp.shape[0] , 3):
            temp[i:i+3 , i:i+3] = np.eye(3)
        R = torch.from_numpy(convert_mat( temp, ncams).transpose(2,0,1))
        gt = ground_truth.detach().clone()
        E_mean , E_median , E_var = compare_rot_graph(R , gt ,method = method)
        return (E_mean , E_median , E_median)


        
    @FLAGS.inject
    def get_d_e2e(self, e2e, shape):
        d_e2e = torch.zeros(shape, device=device).type(torch.float64)
        d_e2e[self.us, self.vs] = self.ys - self.ys_
        d_e2e = d_e2e / len(self.ys_)
        return d_e2e

    @FLAGS.inject
    def get_cvx_opt_constraints(self, x, shape):
        A = np.zeros(shape)
        mask = np.zeros(shape)
        A[self.us.cpu(), self.vs.cpu()] = self.ys_.cpu()
        mask[self.us.cpu(), self.vs.cpu()] = 1
        eps = 1.e-3
        constraints = [cvx.abs(cvx.multiply(x - A, mask)) <= eps]
        return constraints







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
    elif FLAGS.optimizer == 'cvxpy':
        cvx_opt(prob)
        return
    else:
        raise ValueError
    #run_name = f"{FLAGS.dataset}_LR_{FLAGS.lr}_opt_{FLAGS.optimizer}_init_{FLAGS.initialization}_depth_{FLAGS.depth}_scale_{FLAGS.init_scale}"
    wandb.login(key="***REMOVED***")
    run = wandb.init(project=FLAGS.project_name)
    wandb.config.lr = FLAGS.lr
    wandb.config.loss_fn=FLAGS.loss_fn
    wandb.config.optimizer = FLAGS.optimizer
    wandb.config.initialization = FLAGS.initialization
    wandb.config.init_scale = FLAGS.init_scale
    wandb.config.depth = FLAGS.depth
    wandb.config.dataset = FLAGS.dataset
    wandb.config.loss_fn = FLAGS.loss_fn
    wandb.config.act = "None"
    wandb.config.cameras = FLAGS.cameras
    wandb.config.outlier_probability = FLAGS.outlier_probabilty
    wandb.config.fraction_missing = FLAGS.fraction_missing
    wandb.config.reg_lambda = FLAGS.add_reg
   
    log_path =  _fs.resolve("$LOGDIR")
    new_file_dir = os.path.join(log_path , "X_output.mat")
    config_path = os.path.join(log_path, "config.toml")
    with open(config_path) as f:
        lines = f.readlines()
    dataset = lines[3].strip('\n').split("= ")[1].replace('"', '')
    depth = int(lines[4].strip('\n').split("= ")[1])
    init_scale = float(lines[8].strip('\n').split("= ")[1])
    optimizer_ = lines[12].strip('\n').split("= ")[1].replace('"', '')
    initialization  = lines[13].strip('\n').split("= ")[1].replace('"','')
    lr = float(lines[14].strip('\n').split("= ")[1])
    delta = float(lines[11].strip('\n').split("= ")[1])
    config_dic = {"depth":depth , "init_scale": init_scale , "optimizer":optimizer_ , "initialization":initialization , "lr":lr, "dataset":dataset , "delta":delta}

    json_path = os.path.join(log_path, "config.json")
    with open(json_path, 'w') as fp:
        json.dump(config_dic, fp)
  
    config = wandb.config
    init_model(model)
    wandb.watch(model)
    loss = None
    alpha = torch.Tensor([0]).to(device)
    scale = torch.Tensor([0.1]).to(device)

    best_E_mean  = 999999
    best_E_var = 0
    best_E_med = 0 
    # criterion = torch.nn.HuberLoss(delta = 0.5)
    if FLAGS.loss_fn == "l1":
        criterion = torch.nn.L1Loss()
    else:
        criterion = None
    
    
    ground_truth = torch.from_numpy(scipy.io.loadmat(os.path.join("./MATLAB_SO3/datasets_matrices/", dataset+".mat"))['R_gt_c'].transpose(2,0,1))
    ncams = scipy.io.loadmat(os.path.join( "./MATLAB_SO3/datasets_matrices/", dataset+".mat"))['ncams_c'][0][0]
    method = "median"
    
    for T in range(n_iters):
        # return get_e2e(self.net,self.input+t.randn(self.input.shape).cuda()*1e-2)
        e2e = get_e2e(model)
        #e2e = get_e2e_rand(model, prob.incomp_mat +torch.randn(prob.incomp_mat.shape).to(device)*1e-2)

        loss, residual_train = prob.get_train_loss(e2e, alpha = alpha , scale = scale, criterion = criterion )
        
        optimizer.zero_grad()
        loss.backward()

        wandb.log({"train_loss":loss.item()})
        with torch.no_grad():
            test_loss , residual_test= prob.get_test_loss(e2e, alpha = alpha , scale = scale, criterion = criterion)
            unobs_loss = prob.unobserved_loss(e2e , criterion)
           

            if T % FLAGS.n_dev_iters == 0 or loss.item() <= train_thres:

                if isinstance(optimizer, GroupRMSprop):
                    adjusted_lr = optimizer.param_groups[0]['adjusted_lr']
                else:
                    adjusted_lr = optimizer.param_groups[0]['lr']

                E_mean , E_median , E_var = prob.get_eval_loss(e2e, ground_truth, ncams , method=method)

                _log.info(f"Iter #{T}: train = {loss.item():.3e}, test = {test_loss.item():.3e}, Mean = {E_mean}, Median = {E_median}, Var = {E_var}, lr = {adjusted_lr}")

                _writer.add_scalar('loss/train', loss.item(), global_step=T)
                _writer.add_scalar('loss/test', test_loss, global_step=T)
                _writer.add_scalar('loss/E_Mean', E_mean, global_step=T)
                _writer.add_scalar('loss/E_Median', E_median, global_step=T)
                _writer.add_scalar('loss/E_Var', E_var, global_step=T)
                _writer.add_scalar('loss/Unobs_Loss', unobs_loss, global_step=T)

                wandb.log({"test_loss":test_loss.item()})
                wandb.log({"lr":adjusted_lr})
                wandb.log({"E_Mean":E_mean})
                wandb.log({"E_Median":E_median})
                wandb.log({"E_Var":E_var})
                wandb.log({"Unobs_Loss":unobs_loss})
                

            

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
    run.summary["E_Mean"] = best_E_mean
    run.summary["E_Median"] = best_E_med
    run.summary["E_Var"] = best_E_var

   
 
   
   
    run.finish()



if __name__ == '__main__':
    main()
