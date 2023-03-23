
function [err_R] = error_R(R1,R2,single_mean)

% R1 = estimated absolute rotations
% R2 = ground truth absolute rotations
% Computes the optimal rotation, translation and scale necessary to map
% R1 to R2, and evaluates the distance between R1 and R2
%
% Author: Federica Arrigoni, 2015


goods = isfinite(R1(1,1,:)) & isfinite(R2(1,1,:));
R1=R1(:,:,goods(:)); R2=R2(:,:,goods(:));


ncams=size(R1,3);

if nargin < 3
    single_mean=1;
end
    

% compute the optimal rotation that maps R1 to R2 by applying L1 single
% rotation averaging (Weiszfeld algorithm)


R_estimates=zeros(3,3,ncams);
for i=1:ncams
    R_estimates(:,:,i)=R1(:,:,i)'*R2(:,:,i);
    
end


if single_mean==1
    iter_max=20;
    Rmean = L1_single_averaging(R_estimates,iter_max);
elseif single_mean==2
    Rmean = chordal_L2_single_averaging(R_estimates);
else
    error('The possibilities are 1 or 2')
end


% transform the rotations, evaluate the error and update camera centres
err_R=zeros(1,ncams);

for i=1:ncams
    R1(:,:,i)=R1(:,:,i)*Rmean;
    err_R(i)=phi_6(R1(:,:,i),R2(:,:,i))*180/pi;
    
    
    %err_R(i) = norm(R1(:,:,i)-R2(:,:,i),'fro');
end


end





