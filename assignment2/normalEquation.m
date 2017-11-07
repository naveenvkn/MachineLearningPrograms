function weights = normalEquation(lambda,X,y)
% m: the size, i.e., the number of training samples
m = length(X(:,1)); % the length of the first column of X

eX=[ones(m,1) X]

syms v0 v1 v2 v3

w=[v0; v1; v2; v3];
w1 = [v1;v2;v3];
size(eX(1,:))

for l=1:m, hX(l) = eX(l,:)*w; end

Jw = sum((hX - y').^2) + (lambda/2)*(w1'*w1);
 
grad=gradient(Jw)

size(grad)

% Use Matlab function 'solve' to solve the equationa obtained by setting 
% gradient = 0 
[v0, v1, v2, v3]=solve(grad(1), grad(2), grad(3), grad(4));

weights = [];
weights(1,1) = v0;
weights(1,2) = v1;
weights(1,3) = v2;
weights(1,4) = v3;
% Inspect the solutions
end