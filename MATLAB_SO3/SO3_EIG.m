
function [R]=SO3_EIG(G,A)
%
% [R,T]=SO3_EIG(G,A,ncams)
% 
% The absolute orientations of the cameras are computed by using Eigenvalue
% Decomposition. 
% 
% OUTPUT
% R = rotation matrices of the cameras (3 x 3 x ncams)
%
% INPUT
% G = block-matrix containing the relative rotations (3*ncams x 3*ncams)
% A = adiacence matrix corresponding to the image graph
% 
% Reference: "Global Motion Estimation from Point Matches", 2012
% Author: Federica Arrigoni, 2013

ncams=size(A,1);

% G = G.*kron(A,ones(3));
G = G.*repelem(A,3,3);

% d(i) = number of relative rotations available in the i-th row-block
% d=sum(A,2); 
% 
% D=zeros(3*ncams);
% for i=1:ncams
%     D(3*i-2,3*i-2)=1/d(i);
%     D(3*i-1,3*i-1)=1/d(i);
%     D(3*i,3*i)=1/d(i);
% end
D = kron(diag(1./sum(A,2)),eye(3));

% Compute the three leading eigenvectors of inv(D)*G and concatenate them 
% to form the matrix M
% REMARK: the eigenvalues are real since G is symmetric
[M,~]=eigs(D*G,3); 
% [M,~]=eigs(G,3); 


% disp('funzione costo SO(3)')
% M=M*sqrt(ncams);
% cost_rotations=norm((G-M*M'),'fro')^2
% save cost_rotations.mat cost_rotations M

% Projection onto SO(3)
R=zeros(3,3,ncams);
for i=1:1
    % Mi=M(3*i-2:3*i,:) is an estimate of the i-th camera rotation; 
    % due to the relaxation,
    % it is not guarantedd to be a rotation matrix
    [U,~,V] = svd(M(3*i-2:3*i,:))
    R(:,:,i)=U*V'; % nearest orthogonal matrix
    if (det(R(:,:,i))<0)
        R(:,:,i)=-U*V';
    end
end


end

