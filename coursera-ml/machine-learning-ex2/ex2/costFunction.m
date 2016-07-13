function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% h_theta = @(x) sigmoid(theta' * x');

% total = 0;
% for iter = 1:m,
%     total += -y(iter) * log(h_theta(X(iter, :))) - (1 - y(iter)) * log(1 - h_theta(X(iter, :)));
% endfor

% J = total / m;

% t = size(theta);
% for theta_iter = 1:t
%     % for each theta, update it
%     % first sum up all differences
%     int_sum = 0;
%     % loop through all training examples m
%     for inner_iter = 1:m,
%         % sum up the difference
%         int_sum += (h_theta(X(inner_iter, :)) - y(inner_iter)) * X(inner_iter, theta_iter);
%     end

%     grad(theta_iter) = int_sum ./ m;
% end

h_theta = @(x) sigmoid(x * theta);

J = sum( -y' * log(h_theta(X)) - (1 - y)' * log(1 - h_theta(X))) / m;

grad = X' * (h_theta(X) - y) / m;


% =============================================================

end
