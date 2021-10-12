
clear,clc
addpath(genpath("./utils"))
% data_name = 'Alamo';
data_name = 'Ellis_Island';
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

%% Load data

filename=['./datasets_matrices/' data_name];
data = load(filename)
Ac = data.Ac;
ncams_c = data.ncams_c;
% X_output = data.Z_correct  ; 
% X_output = X_output*X_output';
R_gt_c = data.R_gt_c;
path = "/home/ekrivosheev/gnn/deep_matrix_factorization/logs" ; 
files =  dir(path) ; 
dirFlags  = [files.isdir]   ;
subFolders = files(dirFlags) ; 

for idx = 3:length(subFolders)
    filepath = fullfile(path , subFolders(idx).name , "X_output.mat"); 
    X_output = load(filepath).data;
    json_filepath = fullfile(path , subFolders(idx).name , "config.json") ;
    fid = fopen(json_filepath); 
    raw = fread(fid,inf); 
    str = char(raw'); 
    fclose(fid); 
    val = jsondecode(str);
    % NOTE: you need to save output (complete) matrix with name X_output and load it

    %% Compute rotations from completed matrix

    R_our=zeros(3,3,ncams_c);
    % [a,cam]=max(sum(Ac)); % node with highest degree is chosen
    % for k=1:ncams_c
    %     Rk=X_output(3*k-2:3*k,3*cam-2:3*cam);
    %     [U,~,V] = svd(full(Rk));
    %     Rk=U*diag([1 1 det(U*V')])*V'; % nearest orthogonal matrix
    %     R_our(:,:,k)=Rk;
    % end

    % % alternative 

    X_output = double(X_output) ; 
    [R_our]=SO3_EIG(X_output,ones(ncams_c));

    [err_our] = error_R(R_our,R_gt_c);

    disp('Error [degrees] on pairwise rotations:')
    disp(['Mean error = ',num2str( mean(err_our) )])
    disp(['Median error = ',num2str( median(err_our) )])
    val.mean_error = mean(err_our) ;
    val.median_error = median(err_our) ; 
    saveJSONfile(val , json_filepath) ; 
end

