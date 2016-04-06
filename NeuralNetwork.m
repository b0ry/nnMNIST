%% Initialisation
clear ; 
close all; 
clc

inputSize  = 784;   % 28x28 Input Images of Digits
hiddenSize = 50;    % 50 hidden units. We can get more accurate results with more hidden units but 50 is enough for demo purposes.
num_labels = 10;    % 10 labels, 0 mapped to '10' for ease of classification later
lambda = 1e-1;      % regularization parameter
initial_Theta1 = randInitWeights(inputSize, hiddenSize);
initial_Theta2 = randInitWeights(hiddenSize, hiddenSize);
initial_Theta3 = randInitWeights(hiddenSize, num_labels);

% Unroll parameters
initial_params = [initial_Theta1(:) ; initial_Theta2(:); initial_Theta3(:)];

%% Visualise data
%
% The data has been taken from http://cis.jhu.edu/~sachin/digit/digit.html
% If you want to recreate the data add all 10 files found here to the path and run
% createDatasets, else just load.
%
% createDatasets;
%
load('MNIST.mat');
m = size(Xtrain, 1);

% Randomly select 100 data points to display
sel = randperm(size(Xtrain, 1));
sel = sel(1:100);

displayData(Xtrain(sel, :));
mapped_values = reshape(ytrain(sel),10,10)

fprintf('\n<Press any key>\n');
pause;

%% Validation Curve
% Plot error in training and validation sets with different values of
% lambda to find optimal lambda and check for bias/variance.

% Only keeping it here for completion
%validationCurve(Xtrain, ytrain, Xval, yval, inputSize, hiddenSize, num_labels, initial_params);

%% Train Neural Network
fprintf('\nTraining... \n')

options = optimset('MaxIter', 100);

% Create "short hand" for the cost function to be minimized
costFunction = @(p)nnCost(p, inputSize, hiddenSize, num_labels, Xtrain, ytrain, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[params, cost] = fmincg(costFunction, initial_params, options);

% Obtain Thetas back from params
Theta1 = reshape(params(1:hiddenSize * (inputSize + 1)), ...
                 hiddenSize, (inputSize + 1));

Theta2 = reshape(params((1 + (hiddenSize * (inputSize + 1))):(hiddenSize * (inputSize + 1))+ (hiddenSize * (hiddenSize+1))), ...
                 hiddenSize, (hiddenSize + 1));

Theta3 = reshape(params((1 + (hiddenSize * (inputSize + 1))+ (hiddenSize * (hiddenSize+1))):end), ...
                 num_labels, (hiddenSize + 1));

fprintf('\n<Press any key>\n');
pause;

%% New Predictions
pred = predict(Theta1, Theta2, Theta3, Xtrain);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == ytrain)) * 100);

predTest = predict(Theta1, Theta2, Theta3, Xtest);
fprintf('\nTest Set Accuracy: %f\n', mean(double(predTest == ytest)) * 100);

%% Display

%  Randomly permute examples
rp = randperm(size(Xtest,1));

for i = 1:100
    % Display 
    displayData(Xtest(rp(i), :));

    predTest = predict(Theta1, Theta2, Theta3, Xtest(rp(i),:));
    fprintf('\nTest Set Prediction: %d\n', mod(predTest, 10));
    
    % Pause
    fprintf('\n<Press any key>\n');
    pause;
end

