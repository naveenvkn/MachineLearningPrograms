clear all;
data = load('hm2data2.mat');

iterations = 20;
lambdaInterval = 0.5;

% Normalization of data
input_data = data.data;
[m,n]=size(input_data);

muX=mean(input_data);
stdX=std(input_data);

repstd=repmat(stdX,length(input_data),1);
repmu=repmat(muX,length(input_data),1);

standardizedX = (input_data-repmu)./repstd;

[trainInd,testInd] = dividerand(m,0.5,0.5);

XTraining = standardizedX(trainInd,1:3);
YTraining= standardizedX(trainInd, 4);

XTesting = standardizedX(testInd,1:3);
YTesting = standardizedX(testInd,4);

error = zeros(1,iterations);
lambda = 0
lambdaX = zeros(1,iterations);

%Using L2 regularization and finding the lambda which gives min coefficient
minLambda = 0;
minError = intmax;
minWeights = zeros(1,4);
for iteration = 1:iterations
    weights = normalEquation(lambda,XTraining,YTraining)
    m = length(XTesting(:,1));
    eX = [ones(m,1) XTesting];
    hX = zeros(m,1);
    for l=1:m, hX(l,1) = eX(l,:)*weights'; end
    error(1,iteration) = sum((hX - YTesting).^2);
    if(error(1,iteration) < minError)
        minLambda = lambda;
        minError = error(1,iteration);
        minWeights = weights;
    end
    
    lambdaX(1,iteration) = lambda;
    lambda = lambda + lambdaInterval;
end

%finding the minimum coefficient
minWeightCoefficient = 2;
for i = 2:4
    if(abs(minWeights(1,i)) < abs(minWeights(1,minWeightCoefficient)))
        minWeightCoefficient = i;
    end
end

%error value across lambda, we found lambda = 0
plot(lambdaX, error);

%Since we got 3rd column in the input as minumum weight coeffieceint
%we are removing that column
X = standardizedX(:,1:2);
Y= standardizedX(:, 4);

%Using crosssvalind() for k fold cross validation to divide the dataset
%into 3 sets
ind = crossvalind('Kfold', length(standardizedX), 3);
x_error=[1 2 3];
degree1_error = zeros(1,3);
degree2_error = zeros(1,3);
%finding the best polynomial degree which gives minimum error
for i = 1:3
    train = (ind == i); test = ~train;
    test_data = X(test, :); train_data = X(train, :);
    test_data_output = Y(test, :); train_data_output = Y(train, :);
    
    weights = degree1_equation(train_data,train_data_output);
    m = length(test_data(:,1));
    eX = [ones(m,1) test_data];
    hX = zeros(m,1);
    for l=1:m, hX(l,1) = eX(l,:)*weights'; end
    degree1_error(i) = sum((hX - test_data_output).^2);
    
    weights = degree2_equation(train_data,train_data_output);
    m = length(test_data(:,1));
    X1 = test_data(:,1);
    X2 = test_data(:,2);
    X1X2 = zeros(m, 1);
    for l=1:m, X1X2(l) = X1(l,1)*X2(l,1); end
    eX=[ones(m,1) test_data X1.^2 X2.^2 X1X2];
    hX = zeros(m,1);
    for l=1:m, hX(l,1) = eX(l,:)*weights'; end
    degree2_error(i) = sum((hX - test_data_output).^2);
end

%plotting degree1 and degree2 errors
figure(1);
plot(x_error,degree1_error);
disp(degree1_error);
hold on
plot(x_error, degree2_error);
disp(degree2_error);

% From the above error values of degree1 and degree2 equations 
% We got degree2 as best fit. Hence using the same for further calcualtions
modelingError = zeros(1,100);
generalizationError = zeros(1,100);
[data_length,n]=size(input_data);
xAxis = zeros(1,100);

%After L2 regularization, removed 3rd column and taking degree 2 equation
%since it result in min error, We are going to train and test using both
%for 100 iterations
for i = 1:100
    xAxis(1,i) = i;
    [trainInd,testInd] = dividerand(data_length,0.5,0.5);
    train_data = standardizedX(trainInd,1:2);
    train_data_output= standardizedX(trainInd, 4);

    test_data = standardizedX(testInd,1:2);
    test_data_output = standardizedX(testInd,4);

    %calculating modeling error
    weights = degree2_equation(train_data,train_data_output);
    X1 = train_data(:,1); 
    X2 = train_data(:,2);
    m = length(train_data(:,1));
    X1X2 = zeros(m, 1);
    for l=1:m, X1X2(l) = X1(l,1)*X2(l,1); end
    eX=[ones(m,1) train_data X1.^2 X2.^2 X1X2];
    hX = zeros(m,1);
    for l=1:m, hX(l,1) = eX(l,:)*weights'; end
    modelingError(i) = sum((hX - train_data_output).^2);
    
    %calculating generalization error
    X1 = test_data(:,1); 
    X2 = test_data(:,2);
    m = length(test_data(:,1));
    X1X2 = zeros(m, 1);
    for l=1:m, X1X2(l) = X1(l,1)*X2(l,1); end
    eX=[ones(m,1) test_data X1.^2 X2.^2 X1X2];
    hX = zeros(m,1);
    for l=1:m, hX(l,1) = eX(l,:)*weights'; end
    generalizationError(i) = sum((hX - test_data_output).^2);
end


%Below are the Min, max and Average of Modleing Error and Generalization
%error after training the model for 100 iterations
figure(2);
disp('Min ModleingError:');
disp (min(modelingError));
disp('Max ModleingError:');
disp (max(modelingError));
disp('Average ModleingError:');
disp (sum(modelingError)/100);

disp('Min generalizationError:');
disp (min(generalizationError));
disp('Max generalizationError:');
disp (max(generalizationError));
disp('Average generalizationError:');
disp (sum(generalizationError)/100);

legend('generalization', 'modeling')
plot(xAxis, generalizationError);
hold on;
plot(xAxis, modelingError);

x = [0 , 10];
y1 = [0 , weights(2)*10];
y2 = [0, weights(3)*10];

%plotting regression for attributes x1 and x2
figure(3);
plot(x,y1);
hold on 
plot(x,y2);


%plotting contour
W2_vals = linspace(-10, 10, 100);
W3_vals = linspace(-1, 4, 100);
J_vals = zeros(length(W2_vals), length(W3_vals));
for i = 1:length(W2_vals)
     for j = 1:length(W3_vals)
 	  t = [W2_vals(i); W3_vals(j)];    
 	  J_vals(i,j) = computeCostB(X, Y, t);
     end
end

figure(4)
contour(W2_vals, W3_vals, J_vals, logspace(-3, 3, 20))
