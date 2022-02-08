
% clear,clc
addpath(genpath("./utils")) 
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

%% Load data


path = "/home/ekrivosheev/gnn/deep_matrix_factorization/test/11/" ; 


    
filepath = fullfile(path , "X_output.mat"); 
X_output = load(filepath).best;
json_filepath = fullfile(path , "config.json") ;
fid = fopen(json_filepath);
raw = fread(fid,inf); 
str = char(raw'); 
fclose(fid); 
val = jsondecode(str);
data_name = val.dataset ; 
filename = ['./datasets_matrices/', append(data_name,'.mat')]
data = load(filename)
Ac = data.Ac;
ncams_c = data.ncams_c;
R_gt_c = data.R_gt_c;

% % NOTE: you need to save output (complete) matrix with name X_output and load it

% %% Compute rotations from completed matrix

R_our=zeros(3,3,ncams_c);
%[a,cam]=max(sum(Ac)); % node with highest degree is chosen
%for k=1:ncams_c
%     Rk=X_output(3*k-2:3*k,3*cam-2:3*cam);
%    [U,~,V] = svd(full(Rk));
%     Rk=U*diag([1 1 det(U*V')])*V'; % nearest orthogonal matrix
%     R_our(:,:,k)=Rk;
%end 
% % alternative 

X_output = double(X_output) ; 
[R_our]=SO3_EIG(X_output,ones(ncams_c));
save('/home/ekrivosheev/gnn/msp_rot_avg/R1.mat','R_our');

[err_our] = error_R(R_our,R_gt_c);

disp('Error [degrees] on pairwise rotations:')
disp(['Mean error = ',num2str( mean(err_our) )])
disp(['Median error = ',num2str( median(err_our) )])
val.mean_error = mean(err_our) ;
val.median_error = median(err_our) ; 
saveJSONfile(val , json_filepath) ; 
