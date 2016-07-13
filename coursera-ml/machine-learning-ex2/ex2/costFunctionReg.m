function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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


% h_theta = @(x) sigmoid(theta' * x');

% total = 0;

% for iter = 1:m,
%     total += -y(iter) * log(h_theta(X(iter, :))) - (1 - y(iter)) * log(1 - h_theta(X(iter, :)));
% endfor

% J = total / m + sum(theta(2:end).^2) * lambda / (2 * m);

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

%     if theta_iter > 1
%       int_sum += lambda * theta(theta_iter)
%     endif

%     grad(theta_iter) = int_sum ./ m;
% end

h_theta = @(x) sigmoid(x * theta);

temp = theta;
temp(1) = 0;

J = sum( -y' * log(h_theta(X)) - (1 - y)' * log(1 - h_theta(X))) / m + sum(temp .^2) * lambda / (2 * m);

grad = (X' * (h_theta(X) - y) + lambda .* temp)/ m;



% =============================================================

end
