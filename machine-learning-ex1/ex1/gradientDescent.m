function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
      
    x1 = theta(1)*X(:,1);
    x2 = theta(2)*X(:,2);
    hx = x1+x2;
    
    temp0 = hx-y;    
    s0 =  sum(temp0);
    theta_0 = (alpha/m)* s0;  
    
    temp1 = temp0' * X(:,2);   
    theta_1 = (alpha/m)* temp1;
    theta_new = [theta_0 ; theta_1];
    
    theta = theta - theta_new; 
    
    %theta
    % ============================================================
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);        

end

end
