function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h = X*theta;
% size(h)
theta_reg = theta(2:end);
% size(h-y)
J1 = (sum((h - y).^2))/(2*m);


reg = lambda * sum(theta_reg.^2)/(2*m);
J = J1 +reg;



r = size(theta,1);
temp = (h-y);
grad_1(1:r) = (temp'*X(:,1:r))/m;
temp_1(1) = 0;

temp_1(2:r) = lambda*theta_reg/m;
grad(1:r) = grad_1(1:r) + temp_1(1:r);






% =========================================================================

grad = grad(:);

end
