
clc,clear

%% Import ground truth data (Bundler)

f=fopen('./datasets_snavely/Quad/artsquad.iba.out');

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


%% Create ground truth absolute rotations

% R_true_full(:,:,i) is the absolute rotation of the i-th camera
R_true_full=nan(3,3,n);

for i=1:n
    
    Ri=ground_truth(5*i-3:5*i-1,:);
    R_true_full(:,:,i)=Ri;
    
    if all(all(Ri==0)) % insert NaN if the i-th camera rotation is not available
        R_true_full(:,:,i)=NaN(3,3);
    end
    
end


%% Import input relative rotations

f=fopen('./datasets_snavely/Quad/pairs.txt'); % open file

% read the first row (it has 2 entries)
c=fscanf(f,'%u',[2 1]);
ncams=c(1);
npairs=c(2);

% read the remaining npairs rows (they have 18 entries)
[data]=fscanf(f,'%f',[18 npairs]); 
% Transpose so that data matches the orientation of the original file
data=data';

fclose(f); % close file

assert(n==ncams)


%% Compare input and ground truth relative rotations by using the angular distance (range [0,180] degrees)

E=sparse(ncams,ncams); % error matrix

G=eye(3*ncams,3*ncams); % block-matrix containing the relative rotations
A=eye(ncams); % adiacence matrix

for p=1:npairs
    
    % consider the camera pair [i,j]
    i=data(p,1)+1;
    j=data(p,2)+1;
    
    Rij=[data(p,3) data(p,4) data(p,5);data(p,6) data(p,7) data(p,8);data(p,9) data(p,10) data(p,11)];
    
    if isfinite(R_true_full(1,1,i)) &&  isfinite(R_true_full(1,1,j)) % if the absolute rotations i and j are available...
        E(i,j) = phi_6(Rij,  R_true_full(:,:,i)*R_true_full(:,:,j)')*180/pi;
    end
    
    G(3*i-2:3*i,3*j-2:3*j)=Rij;
    G(3*j-2:3*j,3*i-2:3*i)=Rij';
    
    A(i,j)=1;
    A(j,i)=1;
    
end


%% compute the histogram and median error (considering the available relative rotations only)

[~,~,S]=find(E);
figure, hist(S,50)
set(gca,'FontSize',14)
xlabel('Angular Error (degrees)')
title('Histogram - Error on the relative rotations')
disp(['Median Angular Error (degrees): ',num2str(median(S))])
disp(['Mean Angular Error (degrees): ',num2str(mean(S))])

%% Consider only cameras with ground-truth

goods=find(isfinite(R_true_full(1,1,:)));
goods=goods(:);

R_gt_c=R_true_full(:,:,goods);

Ac=A(goods,:); 
Ac=Ac(:,goods);
Ac=sparse(Ac);

ind_keep=sort([3*goods'-2 3*goods'-1 3*goods'],'ascend');
Gc=G(ind_keep,:);
Gc=Gc(:,ind_keep);


%% consider the largest connected component only

[n_conn,C] = graphconncomp(sparse(Ac),'Directed', false);

if n_conn ~=1 % if there are more than 1 connected components
    fprintf('The epipolar graph is not connected: the largest connected component is considered only!\n');
    
    big_comp = mode(C); % consider the largest connected component
    
    index_images=find(C==big_comp); % images that survive
    index=find(C~=big_comp); % images that are eliminated
    
    
    % update A
    Ac(index,:)=[];
    Ac(:,index)=[];
    
    % update ncams, npairs
    ncams=length(index_images);
    npairs=(nnz(Ac)-ncams)/2;
    
    fraction_missing=1-((sum(sum(Ac))-ncams)/2)/nchoosek(ncams,2);
    fprintf('Percentage of missing pairs: %f %%\n',fraction_missing*100)
    
    % update G
    index_G=[3*index-2 3*index-1 3*index];
    Gc(index_G,:)=[];
    Gc(:,index_G)=[];
    
    R_gt_c=R_gt_c(:,:,index_images);
    
end

Gc=sparse(Gc);

conncomp=graphconncomp(Ac)


%% Apply EIG

tic
[R_eig]=SO3_EIG(Gc,Ac);
time=toc

[err_eig]=error_R(R_eig,R_gt_c,1);

median(err_eig)
mean(err_eig)

figure,hist(err_eig,50)
figure,plot(err_eig)


%% write in a file ground-truth rotations 

f=fopen('1DSfMdata_txt/Quad.txt','w');

[I,J]=find(triu(Ac,1));
npairs_c=length(I);

fraction_pairs=npairs_c/nchoosek(ncams,2)*100

% number of nodes / edges
fprintf(f,'%d %d\n',ncams,npairs_c);

[nn,root]=max(sum(Ac));
R_ref=R_gt_c(:,:,root); % set reference frame

for i=1:ncams
    Ri=R_gt_c(:,:,i)*R_ref';
    %qi = iquat(Ri); % My function
    qi=full(R2q(Ri)); % neurora function
    fprintf(f,'%d %f %f %f %f\n',i-1,qi(1),qi(2),qi(3),qi(4)); % 0-based indec
end


% write in a file input rotations + error

for k=1:npairs_c
    i=I(k); j=J(k);
    Rij=Gc(3*j-2:3*j,3*i-2:3*i); % invert notation (i,j) -> (j,i)
    %qij = iquat(Rij);
    qij=full(R2q(Rij)); % neurora function
    
    err(k)=phi_6(R_gt_c(:,:,j)*R_gt_c(:,:,i)',Rij)*180/pi;
    fprintf(f,'%d %d %f %f %f %f %f\n',i-1,j-1,qij(1),qij(2),qij(3),qij(4),err(k)); % 0-based index
end

fclose(f);

figure,hist(err,20)
set(gca,'FontSize',16,'LineWidth',3)
grid on





