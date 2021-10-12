function [R_single] = chordal_L2_single_averaging(R_estimates)
% Performes single rotation averaging (using L2 Frobenius norm)

Ce = sum(R_estimates,3);
[U,~,V] = svd(Ce);
R_single = U*V';

if det(R_single)<0
    R_single=U*diag([1,1,-1])*V';
end

end