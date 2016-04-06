function p = predict(Theta1, Theta2, Theta3, X)

% Number of training examples
m = size(X, 1);
p = zeros(m, 1);

h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
h3 = sigmoid([ones(m, 1) h2] * Theta3');
[~, p] = max(h3, [], 2);

end
