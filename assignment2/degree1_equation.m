function weights = degree1_equation(X,y)
% m: the size, i.e., the number of training samples
m = length(X(:,1)); % the length of the first column of X

eX=[ones(m,1) X];

syms v0 v1 v2

w=[v0; v1; v2];

for l=1:m, hX(l) = eX(l,:)*w; end

Jw = sum((hX - y').^2) ;
 
grad=gradient(Jw);

% Use Matlab function 'solve' to solve the equationa obtained by setting 
% gradient = 0 
[v0, v1, v2]=solve(grad(1), grad(2), grad(3));

weights = [];
weights(1,1) = v0;
weights(1,2) = v1;
weights(1,3) = v2;
% Inspect the solutions
end