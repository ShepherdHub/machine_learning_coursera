function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

Y = eye(num_labels)(y,:); %5000x10

X = [ones(rows(X),1) X]; %5000x401
z2 = X * Theta1'; %5000x25
a2 = sigmoid(z2); % 5000x25

a2 = [ones(rows(a2),1) a2]; %5000x26
z3 = a2 * Theta2'; %5000x10
hypothesis = sigmoid(z3); %5000x10

J = (1/m) * sum(sum(-Y .* log(hypothesis) - (1 - Y) .* log(1-hypothesis)));

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

% Theta1 => 25 x 401
% Theta2 => 10 x 26

%Vectorized implementation
d3 = hypothesis - Y; %5000x10
d2 = d3 * Theta2(:,2:end) .* sigmoidGradient(z2); %5000x10 * 10x25 .* 5000x25

Delta1 = d2' * X; %25x401
Delta2 = d3' * a2; %10x26

Theta1_grad = (1/m) * Delta1;
Theta2_grad = (1/m) * Delta2;

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


theta_reg1 = [zeros(rows(Theta1),1) Theta1(:,2:columns(Theta1))];
theta_reg2 = [zeros(rows(Theta2),1) Theta2(:,2:columns(Theta2))];
reg = (lambda / (2 * m)) * (sum(sum(theta_reg1.^2)) ...
        + sum(sum(theta_reg2.^2)));

J += reg;

back_reg1 = (lambda/m) * theta_reg1;
back_reg2 = (lambda/m) * theta_reg2;

Theta1_grad += back_reg1;
Theta2_grad += back_reg2;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
