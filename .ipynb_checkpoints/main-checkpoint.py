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
import robust_loss_pytorch.general

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
            weight = F.softsign(weight)
        else:
            weight = fc(weight)            
    return F.softsign(weight)


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
        for a, b in zip(matrices, matrices[1:]):
            assert np.allclose(a.dot(a.T), b.T.dot(b), atol=1e-6)
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
        assert 0.8 <= e2e_fro / desired_fro <= 1.2
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
    def __init__(self, *, gt_path, obs_path):
        self.w_gt = torch.load(gt_path, map_location=device)
        (self.us, self.vs), self.ys_ = torch.load(obs_path, map_location=device)

    def get_train_loss(self, e2e,alpha = None , scale = None ,  criterion=None):
        self.ys = e2e[self.us, self.vs]
        residual = (self.ys - self.ys_).type(torch.float32)
        # loss = torch.mean(robust_loss_pytorch.general.lossfun(residual,alpha =alpha  , scale = scale)) ##Cauchy
        loss = criterion(self.ys.to(device).float() , self.ys_.float()) ##Huber
        # loss = (self.ys.to(device) - self.ys_).pow(2).mean() ## L2
        residual = (self.ys.to(device) - self.ys_)
        

        return (loss , residual)

    def get_test_loss(self, e2e,alpha = None  , scale = None ,criterion=None):
        loss = criterion(self.w_gt , e2e.to(device)) ## Huber
        # residual =(self.w_gt  - e2e).type(torch.float32)
        # loss = torch.mean(robust_loss_pytorch.general.lossfun(residual , alpha= alpha , scale = scale))
        # return loss
        residual = (self.w_gt - e2e.cuda()).reshape(-1)  
        # return (self.w_gt - e2e.cuda()).reshape(-1).pow(2).mean() , residual
        return loss , residual 

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



class MatrixCompletionOld(MatrixCompletion):
    @FLAGS.inject
    def __init__(self, *, obs_path):
        
        self.w_gt, (self.us, self.vs), self.ys_ = torch.load(obs_path, map_location=device)



class MatrixSensing(BaseProblem):
    ys: torch.Tensor

    @FLAGS.inject
    def __init__(self, *, gt_path, obs_path):
        self.w_gt = torch.load(gt_path, map_location=device)
        self.xs, self.ys_ = torch.load(obs_path, map_location=device)
    
    def get_train_loss(self, e2e):
        self.ys = (self.xs * e2e).sum(dim=-1).sum(dim=-1)
        return 

    def get_test_loss(self, e2e):
        return (self.w_gt - e2e).view(-1).pow(2).mean()

    @FLAGS.inject
    def get_d_e2e(self, e2e, shape):
        d_e2e = self.xs.view(-1, *shape) * (self.ys - self.ys_).view(len(self.xs), 1, 1)
        d_e2e = d_e2e.sum(0)
        return d_e2e

    def get_cvx_opt_constraints(self, X):
        eps = 1.e-3
        constraints = []
        for x, y_ in zip(self.xs, self.ys_):
            constraints.append(cvx.abs(cvx.sum(cvx.multiply(X, x)) - y_) <= eps)
        return constraints


class MovieLens100k(BaseProblem):
    ys: torch.Tensor

    @FLAGS.inject
    def __init__(self, *, obs_path, n_train_samples):
        (self.us, self.vs), ys_ = torch.load(obs_path, map_location=device)
        # self.ys_ = (ys_ - ys_.mean()) / ys_.std()hidden_si
        self.ys_ = ys_
        self.n_train_samples = n_train_samples

    def get_train_loss(self, e2e):
        self.ys = e2e[self.us[:self.n_train_samples], self.vs[:self.n_train_samples]]
        return (self.ys - self.ys_[:self.n_train_samples]).pow(2).mean()

    def get_test_loss(self, e2e):
        ys = e2e[self.us[self.n_train_samples:], self.vs[self.n_train_samples:]]
        return (ys - self.ys_[self.n_train_samples:]).pow(2).mean()

    @FLAGS.inject
    def get_d_e2e(self, e2e, *, shape):
        d_e2e = torch.zeros(shape, device=device)
        d_e2e[self.us[:self.n_train_samples], self.vs[:self.n_train_samples]] = \
            self.ys - self.ys_[:self.n_train_samples]
        d_e2e = d_e2e / len(self.ys_)
        return d_e2e

    @FLAGS.inject
    def get_cvx_opt_constraints(self, x, *, shape):
        A = np.zeros(shape)
        mask = np.zeros(shape)
        A[self.us[:self.n_train_samples], self.vs[:self.n_train_samples]] = self.ys_[:self.n_train_samples]
        mask[self.us[:self.n_train_samples], self.vs[:self.n_train_samples]] = 1
        eps = 1.e-3
        constraints = [cvx.abs(cvx.multiply(x - A, mask)) <= eps]
        return constraints


@lz.main(FLAGS)
@FLAGS.inject
def main(*, depth, hidden_sizes, n_iters, problem, train_thres, _seed, _log, _writer, _info, _fs):
    prob: BaseProblem
    if problem == 'matrix-completion':
        prob = MatrixCompletion()
    elif problem == 'matrix-sensing':
        prob = MatrixSensing()
    elif problem == 'ml-100k':
        prob = MovieLens100k()
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
    run_name = f"{FLAGS.dataset}_LR_{FLAGS.lr}_opt_{FLAGS.optimizer}_init_{FLAGS.initialization}_depth_{FLAGS.depth}_scale_{FLAGS.init_scale}"
    wandb.login(key="***REMOVED***")
    run = wandb.init(project="mc-rot" ,name = run_name )
    wandb.config.lr = FLAGS.lr
    wandb.config.optimizer = FLAGS.optimizer
    wandb.config.initialization = FLAGS.initialization
    wandb.config.init_scale = FLAGS.init_scale
    wandb.config.depth = FLAGS.depth
    wandb.config.dataset = FLAGS.dataset

    config = wandb.config
    init_model(model)
    wandb.watch(model)
    loss = None
    alpha = torch.Tensor([0]).to(device)
    scale = torch.Tensor([0.1]).to(device)

    best_loss  = 999999
    criterion = torch.nn.HuberLoss(delta = 0.5)
    # criterion = torch.nn.L1Loss()
    
    
    
    for T in range(n_iters):
        e2e = get_e2e(model)

        loss, residual_train = prob.get_train_loss(e2e, alpha = alpha , scale = scale, criterion = criterion )


        params_norm = 0
        for param in model.parameters():
            params_norm = params_norm + param.pow(2).sum()
        optimizer.zero_grad()
        loss.backward()
        wandb.log({"train_loss":loss.item()})
        with torch.no_grad():
            test_loss , residual_test= prob.get_test_loss(e2e, alpha = alpha , scale = scale, criterion = criterion)

            if T % FLAGS.n_dev_iters == 0 or loss.item() <= train_thres:

                U, singular_values, V = e2e.svd()  # U D V^T = e2e
                schatten_norm = singular_values.pow(2. / depth).sum()

                d_e2e = prob.get_d_e2e(e2e)
                full = U.t().mm(d_e2e.float()).mm(V).abs()  # we only need the magnitude.
                n, m = full.shape

                diag = full.diag()
                mask = torch.ones_like(full, dtype=torch.int)
                mask[np.arange(min(n, m)), np.arange(min(n, m))] = 0
                off_diag = full.masked_select(mask > 0)
                _writer.add_scalar('diag/mean', diag.mean().item(), global_step=T)
                _writer.add_scalar('diag/std', diag.std().item(), global_step=T)
                _writer.add_scalar('off_diag/mean', off_diag.mean().item(), global_step=T)
                _writer.add_scalar('off_diag/std', off_diag.std().item(), global_step=T)

                grads = [param.grad.cpu().data.numpy().reshape(-1) for param in model.parameters()]
                grads = np.concatenate(grads)
                avg_grads_norm = np.sqrt(np.mean(grads**2))
                avg_param_norm = np.sqrt(params_norm.item() / len(grads))

                if isinstance(optimizer, GroupRMSprop):
                    adjusted_lr = optimizer.param_groups[0]['adjusted_lr']
                else:
                    adjusted_lr = optimizer.param_groups[0]['lr']
                _log.info(f"Iter #{T}: train = {loss.item():.3e}, test = {test_loss.item():.3e}, "
                          f"Schatten norm = {schatten_norm:.3e}, "
                          f"grad: {avg_grads_norm:.3e}, "
                          f"lr = {adjusted_lr:.3f}")

                _writer.add_scalar('loss/train', loss.item(), global_step=T)
                _writer.add_scalar('loss/test', test_loss, global_step=T)
                _writer.add_scalar('Schatten_norm', schatten_norm, global_step=T)
                _writer.add_scalar('norm/grads', avg_grads_norm, global_step=T)
                _writer.add_scalar('norm/params', avg_param_norm, global_step=T)
                wandb.log({"test_loss":test_loss.item()})
                wandb.log({"Schatten_Norm":schatten_norm})
                wandb.log({"grad_norm":avg_grads_norm})
                wandb.log({"param_norm":avg_param_norm})
                wandb.log({"lr":adjusted_lr})
                torch.save(residual_test , _fs.resolve("$LOGDIR/res_final_test.pt"))
                torch.save(residual_train , _fs.resolve("$LOGDIR/res_final_train.pt"))
                for i in range(FLAGS.n_singulars_save):
                    _writer.add_scalar(f'singular_values/{i}', singular_values[i], global_step=T)
                if(test_loss.item()<best_loss):
                    torch.save(e2e, _fs.resolve("$LOGDIR/best.npy"))
                    torch.save(residual_test , _fs.resolve("$LOGDIR/res_best_test.pt"))
                    torch.save(residual_train , _fs.resolve("$LOGDIR/res_best_train.pt"))
                    best_loss = test_loss.item()
                
                if loss.item() <= train_thres:
                    break
        optimizer.step()

    _log.info(f"train loss = {loss.item()}. test loss = {test_loss.item()}")
    torch.save(e2e, _fs.resolve("$LOGDIR/final.npy"))


    """Save the matrix in .mat for evaluation later in MATLAB"""
    final_path = _fs.resolve("$LOGDIR/final.npy")
    best_path = _fs.resolve("$LOGDIR/best.npy")
    log_path =  _fs.resolve("$LOGDIR")
    new_file_dir = os.path.join(log_path , "X_output.mat")
    final_mat = torch.load(final_path).detach().cpu().numpy()
    best_mat = torch.load(best_path).detach().cpu().numpy()
    ndic = {'best': best_mat, 'final':final_mat}
    scipy.io.savemat(new_file_dir , ndic)
    config_path = os.path.join(log_path, "config.toml")
    with open(config_path) as f:
        lines = f.readlines()
    dataset = lines[3].strip('\n').split("= ")[1].replace('"', '')
    depth = int(lines[4].strip('\n').split("= ")[1])
    init_scale = float(lines[8].strip('\n').split("= ")[1])
    optimizer = lines[12].strip('\n').split("= ")[1].replace('"', '')
    initialization  = lines[13].strip('\n').split("= ")[1].replace('"','')
    lr = float(lines[14].strip('\n').split("= ")[1])
    delta = float(lines[11].strip('\n').split("= ")[1])
    config_dic = {"depth":depth , "init_scale": init_scale , "optimizer":optimizer , "initialization":initialization , "lr":lr, "dataset":dataset , "delta":delta}
    json_path = os.path.join(log_path, "config.json")
    with open(json_path, 'w') as fp:
        json.dump(config_dic, fp)
    os.remove(final_path)
    os.remove(best_path)
    run.finish()



if __name__ == '__main__':
    main()
