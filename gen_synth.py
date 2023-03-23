import torch
import scipy.io
import os
import glob
import numpy as np


def gen_data():
    base_path = "./datasets/data-synth-mat"
    save_path = "./datasets/data-synth"
    #Iterate through all the files in the folder
    for file in glob.glob(os.path.join(base_path, "*.mat")):
        file_name = os.path.basename(file)
        name = os.path.splitext(file_name)[0]
        mat = scipy.io.loadmat(file)
        Z_incomplete = mat['Z_incomplete']
        complete_matrix_gt = np.matmul(mat['Z_correct'], mat['Z_correct'].T)
        Omega = mat["Omega"]
        data = []
        data_unobs = []
        for i in range(0, Z_incomplete.shape[0], 3):
            for j in range(0, Z_incomplete.shape[0], 3):
                if(Omega[i][j] == 1):
                    data.append([i, j])
                else:
                    data_unobs.append([i , j])
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        final_path = os.path.join(save_path , name+"_gt.pt")
        torch.save(torch.from_numpy(complete_matrix_gt), final_path)
        x =[]
        y = []
        for idx in data:
            for i in range(3):
                for j in range(3):
                    x.append(idx[0]+i)
                    y.append(idx[1]+j)
        x = torch.tensor(x)
        y = torch.tensor(y)
        ys_ = Z_incomplete[x , y]
        ys_ = torch.tensor(ys_)
        obs_path = os.path.join(save_path , name+"_obs.pt")
        torch.save([(x, y), ys_], obs_path)
        x_un =[]
        y_un = []
        for idx in data_unobs:
            for i in range(3):
                for j in range(3):
                    x_un.append(idx[0]+i)
                    y_un.append(idx[1]+j)
        x_un = torch.tensor(x_un)
        y_un = torch.tensor(y_un)
        ys_un = complete_matrix_gt[x_un , y_un]
        ys_un = torch.tensor(ys_un)
        unobs_path = os.path.join(save_path , name+"_unobs.pt")
        torch.save([(x_un, y_un), ys_un], unobs_path)


def gen_config():
    base_path = "datasets/data-synth-mat"
    data_base_path = "datasets/data-synth"
    save_path = "configs/data-synth"
    for file in glob.glob(os.path.join(base_path, "*.mat")):
        file_name = os.path.basename(file)
        name = os.path.splitext(file_name)[0]
        data = []
        data.append("problem = \"matrix-completion\" \n")
        gt_path = os.path.join(data_base_path, name+"_gt.pt")
        data.append(f"gt_path = \"{gt_path}\" \n")
        x = torch.load(gt_path)
        data.append(f"shape = [{x.shape[0]}, {x.shape[0]}] \n")
        obs_path =  os.path.join(data_base_path, name+"_obs.pt")
        data.append(f"obs_path = \"{obs_path}\" \n")
        data.append(f"dataset = \"{name}\" \n")
        unobs_path = os.path.join(data_base_path, name+"_unobs.pt")
        data.append(f"unobs_path = \"{unobs_path}\" \n")
        data.append(f"gt_mat = \"{file}\" \n")
        file_path = os.path.join(save_path , name+".toml")
        file1 = open(file_path,"w")
        file1.writelines(data)
        file1.close()

if __name__ == "__main__":
    gen_data()
    gen_config()