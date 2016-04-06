function [Xtrain, ytrain] = importData

inputSize = 784;
num_labels = 10;
m = 8000;
n = 1000;

X = zeros(10000,inputSize);
Xtrain = zeros(m,inputSize);
Xval = zeros(n,inputSize);
Xtest = zeros(n,inputSize);
ypre = zeros(1000,10);

delta = 1000;
%trainingDelta = m/num_labels;
%testDelta = n/num_labels; 
A = {'data0' 'data1' 'data2' 'data3' 'data4' 'data5' 'data6' 'data7' 'data8' 'data9'};

for k = 1:num_labels 
    fid = fopen(A{k},'r');
    deltaPlus = (k-1)*delta;
    for i = deltaPlus+1:deltaPlus+delta
        X(i,:) = fread(fid,[1,784],'uchar'); 
    end
end

for i = 1:10
    for j = 1:delta
        if i == 1
            ypre(j,i) = 10;
        else
            ypre(j,i) = i-1; 
        end
    end
end

y = ypre(:);

rp = randperm(10000);

X = X(rp,:);
y = y(rp);

Xtrain = X(1:m,:);
%Xval = X(m+1:m+n,:);
%Xtest = X(m+n+1:10000,:);


ytrain = y(1:m);
%yval = y(m+1:m+n);
%ytest = y(m+n+1:10000);

end