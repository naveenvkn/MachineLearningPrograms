function weights = degree2_equation(X,y)
% m: the size, i.e., the number of training samples
m = length(X(:,1)); % the length of the first column of X

% extend the data set by adding a quadratic column
X1 = X(:,1);
X2 = X(:,2);
X1X2 = zeros(m, 1);

for l=1:m, X1X2(l) = X1(l,1)*X2(l,1); end
eX=[ones(m,1) X X1.^2 X2.^2 X1X2];

syms v0 v1 v2 v3 v4 v5

w=[v0; v1; v2; v3; v4; v5];

for l=1:m, hX(l) = eX(l,:)*w; end

Jw = sum((hX - y').^2) ;
 
grad=gradient(Jw);

% Use Matlab function 'solve' to solve the equationa obtained by setting 
% gradient = 0 
[v0, v1, v2, v3, v4, v5]=solve(grad(1), grad(2), grad(3), grad(4), grad(5), grad(6));

weights = [];
weights(1,1) = v0;
weights(1,2) = v1;
weights(1,3) = v2;
weights(1,4) = v3;
weights(1,5) = v4;
weights(1,6) = v5;
% Inspect the solutions
end