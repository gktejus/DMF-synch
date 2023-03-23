

clear,clc,close all
addpath(genpath('./'));

ncams_c = 300; % number of cameras
prob_out=0.3; % fraction of outliers
sigma_a=5; % noise (degrees)
fraction_missing = 0.5; % fraction of missing data


%% generate ground truth

R_gt = zeros(3,3,ncams_c); % absolute rotations
for i = 1:ncams_c
    R_gt(:,:,i) = eul(randn(1,3)); % random Euler angles
end

% 3 x 3*ncams_c matrix with absolute rotations
Y_R = reshape(permute(R_gt,[1,3,2]),[],3); 

% 3*ncams_c x 3*ncams_c matrix with relative rotations
G_gt = Y_R*Y_R'; 


%% add noise on relative rotations

G = G_gt; % noisy matrix with relative rotations
for i=1:ncams_c
    for j=i+1:ncams_c
        
        % random axis
        r = rand(3,1)-0.5;
        if norm(r)~=0
            r=r/norm(r);
        end
        
        % small angle
        angle = randn()+sigma_a;
        angle=angle*pi/180;
        noise = inv_axis_angle(angle,r);
        
        G(3*i-2:3*i,3*j-2:3*j)=G(3*i-2:3*i,3*j-2:3*j)*noise(1:3,1:3);
        G(3*j-2:3*j,3*i-2:3*i)=G(3*i-2:3*i,3*j-2:3*j)';
        
    end
end


%% generate the epipolar graph

if fraction_missing==0
    A=ones(ncams_c);
else
    n_conn=2;
    while n_conn~=1 % generate a new graph if the current one is not connected
        A=rand(ncams_c)>=fraction_missing;
        A=tril(A,-1);
        A=A+A'+eye(ncams_c); % symemtric Adjacency matrix
        [n_conn] = graphconncomp(sparse(A),'Directed', false);
    end
    
end

%% put missing blocks in correspondence of missing pairs

G = G.*kron(A,ones(3));


%% Generate the outlier graph

[I,J]=find(triu(A));
n_pairs=length(I);

n_conn=2;
while n_conn~=1
    
    W=rand(n_pairs,1)<prob_out;
    A_outliers=sparse(I,J,W,ncams_c,ncams_c);
    A_outliers=A_outliers+A_outliers'; % graph representing outlier rotations
    
    % graph representing inlier rotations: it must be connected
    A_inliers=not(A_outliers)&A; 
    [n_conn] = graphconncomp(sparse(A_inliers),'Directed', false);
end


%% Add outliers among relative motions

for i=1:ncams_c
    for j=i+1:ncams_c
        
        if A_outliers(i,j)==1
            
            [u,~,v]=svd(rand(3));
            Rij=u*diag([1 1 det(u*v')])*v';
            
            G(3*i-2:3*i,3*j-2:3*j)=Rij;
            G(3*j-2:3*j,3*i-2:3*i)=Rij';
            
        end
    end
end


%% EIG - synchronization via eigenvalue decomposition

tic
[R]=SO3_EIG(G,A);
time_EIG=toc

[err_R] = error_R(R,R_gt);

disp('Error [degrees] on pairwise rotations:')
disp(['Mean error = ',num2str( mean(err_R) )])
disp(['Median error = ',num2str( median(err_R) )])

%% EIG SO(3) + IRLS (robust synchronization)

% Parameters
cost_function='cauchy';
nmax_IRLS=30;
epsilon_IRLS=1e-5;

tic
[R_IRLS,iter_EIG_IRLS] = SO3_eig_IRLS(G,A,nmax_IRLS,epsilon_IRLS,cost_function,1,0,0);
time_IRLS=toc
[err_IRLS] = error_R(R_IRLS,R_gt);

disp('Error [degrees] on pairwise rotations:')
disp(['Mean error = ',num2str( mean(err_IRLS) )])
disp(['Median error = ',num2str( median(err_IRLS) )])

%%  DEEP MC

% % DMF setup
% s=[5 10 3*ncams_c];% input size, hidden size 1, ..., output size
% options.Wp=0.01;
% options.Zp=0.01;
% options.maxiter=1000;
% options.activation_func={'tanh_opt','linear'};
% [X_DMF,NN_MF]=MC_DMF(G,kron(A,ones(3)),s,options);
% 
% tic
% [R_DMF]=SO3_EIG(X_DMF,ones(ncams_c));
% time_DMF=toc
% 
% [err_DMF] = error_R(R_DMF,R_gt);
% 
% disp('Error [degrees] on pairwise rotations:')
% disp(['Mean error = ',num2str( mean(err_DMF) )])
% disp(['Median error = ',num2str( median(err_DMF) )])


%% save data for Python

Z_incomplete=full(G);
Omega=full(kron(A,ones(3)));
Z_correct=Y_R;

filename = [append('../datasets/data-synth-mat/synth', '_cam_', num2str(ncams_c), '_out_', num2str(prob_out), '_miss_', num2str(fraction_missing), '.mat')];
save(filename, 'Z_incomplete', 'Omega', 'Z_correct', 'A', 'ncams_c', 'R_gt', 'prob_out', 'fraction_missing')



