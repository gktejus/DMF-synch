

clc,clear,close all
addpath(genpath('./'));

run_competitors=false;

% data_name = 'Alamo';
% data_name = 'Ellis_Island';
% data_name = 'Madrid_Metropolis';
% data_name = 'Montreal_Notre_Dame';
% data_name = 'Notre_Dame';
% data_name = 'NYC_Library';
% data_name = 'Piazza_del_Popolo';
% data_name = 'Piccadilly';
% data_name = 'Roman_Forum';
% data_name = 'Tower_of_London';
% data_name = 'Trafalgar';
% data_name = 'Union_Square';
% data_name = 'Vienna_Cathedral';
% data_name = 'Yorkminster';
% data_name = 'Gendarmenmarkt';

datasets = ["Alamo", "Ellis_Island", "Madrid_Metropolis", "Montreal_Notre_Dame", "Notre_Dame", "NYC_Library", "Piazza_del_Popolo", "Piccadilly", "Roman_Forum", "Tower_of_London", "Trafalgar", "Union_Square", "Vienna_Cathedral", "Yorkminster","Gendarmenmarkt"];
%% Import ground truth data (Bundler)

% bundle.out is a reconstruction of approximately the same component of
% the dataset which is described by the other files. Due to differences in
% the reconstruction method, it may have a few extra images, or fail to
% reconstruct some images in the connected component. This reconstruction
% is made with:
% Noah Snavely, Steven M. Seitz, and Richard Szeliski. Photo Tourism:
% Exploring Photo Collections in 3D. SIGGRAPH Conf. Proc., 2006.
for idx = 1:length(datasets)
    data_name = datasets{idx} 
    f=fopen(strcat('./datasets_snavely/datasets/',data_name,'/gt_bundle.out'));
    disp(f)
    % read the first row (it has 18 characters)
    a=fscanf(f,'%c',[18 1]);
    disp(['Ground Truth: ',a']) % Bundle file v0.3

    % read the second row (it has 2 entries: number of cameras and 3D points)
    b=fscanf(f,'%u',[2 1]);
    n=b(1); % total number of images

    % read the remaining rows (they have 3 entries)
    ground_truth=fscanf(f,'%f',[3,5*n]);
    % Transpose so that data matches the orientation of the original file
    ground_truth=ground_truth';

    fclose(f);


    %% Define ground truth motions

    R_gt=nan(3,3,n); % absolute rotations

    cameras_gt=[]; % cameras reconstructed
    for i=1:n
        
        Ri=ground_truth(5*i-3:5*i-1,:);
        Ti=ground_truth(5*i,:)';
        
        if all(all(Ri==0)) % insert NaN if the i-th camera rotation is not available
            R_gt(:,:,i)=NaN(3,3);
            T_gt(:,i)=NaN(3,1);
        else
            R_gt(:,:,i)=Ri;
            T_gt(:,i)=Ti;
            cameras_gt=[cameras_gt i-1];
        end
        
    end


    %% identify images to reconstruct

    % cc.txt: This is a list of camera indices, one per line, specifying which
    % images to reconstruct. These form a single connected component of EGs.

    f=fopen(strcat('./datasets_snavely/datasets/',data_name,'/cc.txt'));
    cc=fscanf(f,'%u',[1,inf]);
    fclose(f);

    cc=sort(cc);
    ncams=length(cc);


    %% Import two-view data

    % EGs.txt: Two-image models are listed in this file, one per line. The
    % format is: `<i> <j> <Rij> <tij>` where i and j are camera indices, Rij is
    % a row-major pairwise rotation matrix, and tij is the position of camera j
    % in camera i's coordinate system. If Ri and Rj are the rotation matrices
    % of cameras i and j (world-to-camera maps) then in the absence of noise
    % Rij = Ri * Rj', ie Rij is the pose of camera j in camera i's coordinate
    % system (where a pose is the transpose of a rotation matrix, a
    % camera-to-world map).
    % All of these EGs are within the connected component.

    f=fopen(strcat('./datasets_snavely/datasets/',data_name,'/EGs.txt'));

    % read all the rows (they have 14 entries)
    [EGs]=fscanf(f,'%f',[14 inf]);
    % Transpose so that data matches the orientation of the original file
    EGs=EGs';
    npairs=size(EGs,1);
    fclose(f); % close file

    G=eye(3*ncams); % block-matrix with pairwise rotations
    A=eye(ncams); % adjacency matrix describing the view-graph
    E_R=nan(ncams); % error on pairwise rotations
    R_gt_reduced=R_gt(:,:,cc+1); % ground-truth rotations on conncected component

    for p=1:npairs
        
        % consider the camera pair [i,j]
        cam_i=EGs(p,1);
        cam_j=EGs(p,2);
        
        [ans_i,i]=find(cc==cam_i);
        [ans_j,j]=find(cc==cam_j);
        
        if ans_i && ans_j
            
            Rij=[EGs(p,3) EGs(p,4) EGs(p,5);EGs(p,6) EGs(p,7) EGs(p,8);EGs(p,9) EGs(p,10) EGs(p,11)];
            
            G(3*i-2:3*i,3*j-2:3*j)=Rij;
            G(3*j-2:3*j,3*i-2:3*i)=Rij';
            
            A(i,j)=1;
            A(j,i)=1;
            
            % evaluate initial errors
            if isfinite(R_gt_reduced(1,1,i)) &&  isfinite(R_gt_reduced(1,1,j))
                E_R(i,j)=phi_6(R_gt_reduced(:,:,i)*R_gt_reduced(:,:,j)',Rij)*180/pi;
            end
            
        end
    end

    G=sparse(G);
    A=sparse(A);

    fraction_missing=(1-(nnz(A)-ncams)/nchoosek(ncams,2)/2)*100;
    disp(['Fraction of missing pairs = ',num2str(fraction_missing),' %'])

    disp('Error [degrees] on pairwise rotations:')
    disp(['Mean error = ',num2str(nanmean(E_R(:)))])
    disp(['Median error = ',num2str(nanmedian(E_R(:)))])

    disp(['Number of cameras = ',num2str(ncams)])

    % figure
    % hist(E_R(:),20)
    % title('Input (pairwise) rotations')
    % xlabel('Error [degrees]')
    % set(gca,'FontSize',16,'LineWidth',3)
    % grid on

    [index,i_cc,i_gt]=intersect(cc,cameras_gt);
    %n_outliers=nnz(E_R(:)>10);
    %n_rotations=nnz(triu(A));
    %n_outliers/n_rotations
    %fraction_inliers=(n_rotations-n_outliers)/nchoosek(ncams,2)

    [~,ind]=max(sum(A)); % camera with most edges

    %% Use only common cameras

    Ac=A(i_cc,:);
    Ac=Ac(:,i_cc);

    conncomp=graphconncomp(Ac)

    ind_keep=sort([3*i_cc'-2 3*i_cc'-1 3*i_cc'],'ascend');
    Gc=G(ind_keep,:);
    Gc=Gc(:,ind_keep);

    R_gt_c=R_gt(:,:,cameras_gt(i_gt)+1);

    ncams_c=length(i_cc)

    %% EIG - synchronization via eigenvalue decomposition

    if run_competitors
        
        tic
        [R]=SO3_EIG(Gc,Ac);
        time_EIG=toc
        
        [err_R] = error_R(R,R_gt_c);
        
        disp('Error [degrees] on pairwise rotations:')
        disp(['Mean error = ',num2str( mean(err_R) )])
        disp(['Median error = ',num2str( median(err_R) )])
        
        %% EIG SO(3) + IRLS (robust synchronization)
        
        % Parameters
        cost_function='cauchy';
        nmax_IRLS=30;
        epsilon_IRLS=1e-5;
        
        tic
        [R_IRLS,iter_EIG_IRLS] = SO3_eig_IRLS(Gc,Ac,nmax_IRLS,epsilon_IRLS,cost_function,1,0,0);
        time_IRLS=toc
        [err_IRLS] = error_R(R_IRLS,R_gt_c);
        
        disp('Error [degrees] on pairwise rotations:')
        disp(['Mean error = ',num2str( mean(err_IRLS) )])
        disp(['Median error = ',num2str( median(err_IRLS) )])
        
        
        %%  DEEP MC
        
        % DMF setup
        s=[3 15 30 3*ncams_c];% input size, hidden size 1, ..., output size
        options.Wp=0.01;
        options.Zp=0.01;
        options.maxiter=100;
        options.activation_func={'tanh_opt','tanh_opt','linear'};
        tic
        [X_DMF,NN_MF]=MC_DMF(Gc,kron(Ac,ones(3)),s,options);
        time_DMF=toc
        
        % Compute rotations from completed matrix
        R_DMF=zeros(3,3,ncams_c);
        [a,cam]=max(sum(Ac)); % node with highest degree is chosen
        for k=1:ncams_c
            Rk=X_DMF(3*k-2:3*k,3*cam-2:3*cam);
            [U,~,V] = svd(full(Rk));
            Rk=U*diag([1 1 det(U*V')])*V'; % nearest orthogonal matrix
            R_DMF(:,:,k)=Rk;
        end
        
        % % alternative
        % [R_DMF]=SO3_EIG(X_DMF,ones(ncams));
        
        [err_DMF] = error_R(R_DMF,R_gt_c);
        
        disp('Error [degrees] on pairwise rotations:')
        disp(['Mean error = ',num2str( mean(err_DMF) )])
        disp(['Median error = ',num2str( median(err_DMF) )])
        
    end

    %% create ground-truth (full) matrix

    X_gt=zeros(3*ncams_c,3);
    for k=1:ncams_c
        Rk=R_gt_c(:,:,k);
        X_gt(3*k-2:3*k,:)=Rk;
    end
    G_gt=X_gt*X_gt';

    %% save data for Python

    Z_incomplete=full(Gc);
    Omega=full(kron(Ac,ones(3)));
    Z_correct=X_gt;

    filename=['./datasets_matrices/' data_name];
    save(filename,'Z_incomplete','Omega','Z_correct','Ac','ncams_c','R_gt_c')
end

