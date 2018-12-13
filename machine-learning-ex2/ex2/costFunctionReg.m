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

x = X*theta;
h = sigmoid(x);
h1 = log(h);
h2 = log(1-h);
j = (y.*h1) + ((1-y).*h2);
j = sum(j);
J = -j/m;

n = size(X,2);

J = J + (lambda/2/m)*sum((theta(2:end)).^2);

grad(1) = sum((h - y).*(X(:,1)))/m;
for j = 2:n
    grad(j) = (sum((h - y).*(X(:,j)))/m) + ((theta(j)*lambda)/m);
end





% =============================================================

end
