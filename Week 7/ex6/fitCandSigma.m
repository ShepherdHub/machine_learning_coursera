%% Initialization
clear ; close all; clc

% Load data
load("ex6data3.mat");

% Create C array
C = [0.01, 0.03, .1, 0.3, 1, 3, 10, 30];
m = length(C);

% Create sigma array
sigma = [0.01, 0.03, .1, 0.3, 1, 3, 10, 30];
n = length(sigma);

% Create empty results matrix zeros(m, n)
results = zeros(m,n);

% Iterate over C (i)
%   Iterate over sigma (j)
%     Train SVM using C(i), sigma(j) on X, y
%     Use trained SVM to predict Xval, yval
%     Calculate error
%     Put error in matrix at (i,j)

for i=1:m
  for j=1:n
    model = svmTrain(X, y, C(i), @(x1, x2) gaussianKernel(x1, x2, sigma(j))); 
    pred = svmPredict(model, Xval);
    results(i,j) = svmError(pred, yval);
  endfor
endfor

% print matrix
results

% Find minimum value and print
[minVal, min_i] = min(min(results,[],2));
[minVal, min_j] = min(min(results,[],1));

printf("The minimum value is %f\n", minVal);
printf("The value of C at minimum is: %f\n", C(min_i));
printf("The value of sigma at minimum is %f\n", sigma(min_j));






