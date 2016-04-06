function validationCurve (Xtrain, ytrain, Xval, yval, inputSize, hiddenSize, num_labels, initial_params)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';
errorT = zeros(length(lambda_vec));
errorV = zeros(length(lambda_vec));

fprintf('\nLambda(Regularisation) Curve... This will take a while. \n')

for i = 1: length(lambda_vec)

    options = optimset('MaxIter', 50);
    costFunction = @(p)nnCost(p, inputSize, hiddenSize, num_labels, Xtrain, ytrain, i);
    [params, ~] = fmincg(costFunction, initial_params, options);
    
    Theta1 = reshape(params(1:hiddenSize * (inputSize + 1)), ...
                 hiddenSize, (inputSize + 1));
    Theta2 = reshape(params((1 + (hiddenSize * (inputSize + 1))):(hiddenSize * (inputSize + 1))+ (hiddenSize * (hiddenSize+1))), ...
                 hiddenSize, (hiddenSize + 1));
    Theta3 = reshape(params((1 + (hiddenSize * (inputSize + 1))+ (hiddenSize * (hiddenSize+1))):end), ...
                 num_labels, (hiddenSize + 1));

    % Calculate errors for training and cross-validation sets
    pred = predict(Theta1, Theta2, Theta3, Xtrain);
    errorT(i) = 1-mean(double(pred == ytrain));

    predVal = predict(Theta1, Theta2, Theta3, Xval);
    errorV(i) = 1-mean(double(predVal == yval));
end
plot(lambda_vec, errorT, lambda_vec, errorV);

fprintf('\n<Press any key>\n');
pause;
end