function [J, grad] = nnCost(params, inputSize, hiddenSize, num_labels, X, y, lambda)

Theta1 = reshape(params(1:hiddenSize * (inputSize + 1)), ...
                 hiddenSize, (inputSize + 1));

Theta2 = reshape(params((1 + (hiddenSize * (inputSize + 1))):(hiddenSize * (inputSize + 1))+ (hiddenSize * (hiddenSize+1))), ...
                 hiddenSize, (hiddenSize + 1));

Theta3 = reshape(params((1 + (hiddenSize * (inputSize + 1))+ (hiddenSize * (hiddenSize+1))):end), ...
                 num_labels, (hiddenSize + 1));

% Number of training examples
m = size(X, 1);

% Initialise cost J and deltas for back-prop.
J = 0;
delta1 = zeros(size(Theta1));
delta2 = zeros(size(Theta2));
delta3 = zeros(size(Theta3));

K = num_labels;
new_y = zeros(m,num_labels);


for i = 1:m
    for k = 1:K
        if y(i) == k
            new_y(i,k) = 1;
        end
    end
    
    % Move through network (and add bias term!)
    a1 = [1 X(i,:)];
    z2 = a1*Theta1';
    a2 = sigmoid(z2);
    a2 = [1 a2];
    z3 = a2*Theta2';
    a3 = sigmoid(z3);
    a3 = [1 a3];
    z4 = a3*Theta3';
    a4 = sigmoid(z4);
    
    % Add to cost function
    J = J + (new_y(i,:)*log(a4')+(1-new_y(i,:))*log(1-a4)');  

    % Back propagation
    d4 = a4-new_y(i,:);
    d3 = d4*Theta3(:,2:end).*sigmoidGradient(z3);
    d2 = d3*Theta2(:,2:end).*sigmoidGradient(z2);
    delta3 = (delta3 + d4'*a3);
    delta2 = (delta2 + d3'*a2); 
    delta1 = (delta1 + d2'*a1); 
end

J = -J./m;

newTheta1 = Theta1(:,2:end).^2;
vecTheta1 = newTheta1(:);
newTheta2 = Theta2(:,2:end).^2;
vecTheta2 = newTheta2(:);
newTheta3 = Theta3(:,2:end).^2;
vecTheta3 = newTheta3(:);
J = J + lambda.*(sum(vecTheta1) + sum(vecTheta2) + sum(vecTheta3))./(2.*m);

Theta1(:,1) = zeros(size(Theta1(:,1)))';
Theta2(:,1) = zeros(size(Theta2(:,1)))';
Theta3(:,1) = zeros(size(Theta3(:,1)))';

Theta1_grad = delta1/m + lambda.*Theta1/m;
Theta2_grad = delta2/m + lambda.*Theta2/m;
Theta3_grad = delta3/m + lambda.*Theta3/m;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:) ; Theta3_grad(:)];

end
