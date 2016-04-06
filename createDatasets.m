function createDatasets

inputSize = 784;
num_labels = 10;
m = 8000;
n = 1000;

Xtrain = zeros(m,inputSize);
Xval = zeros(n,inputSize);
Xtest = zeros(n,inputSize);

ytrain = zeros(m,1);
yval = zeros(n,1);
ytest = zeros(n,1);


deltaTrain = m/num_labels;
deltaVal = n/num_labels;

A = {'data0' 'data1' 'data2' 'data3' 'data4' 'data5' 'data6' 'data7' 'data8' 'data9'};

for k = 1:num_labels 
    fid = fopen(A{k},'r');
    deltaPlus_T = (k-1)*deltaTrain;
    deltaPlus_V = (k-1)*deltaVal;
    for i = deltaPlus_T+1:deltaPlus_T+deltaTrain
        Xtrain(i,:) = fread(fid,[1,784],'uchar'); 
        if k == 1
            ytrain(i) = 10;
        else
            ytrain(i) = k-1;
        end
    end
    for i = deltaPlus_V+1:deltaPlus_V+deltaVal
        Xval(i,:) = fread(fid,[1,784],'uchar'); 
    end
    for i = deltaPlus_V+1:deltaPlus_V+deltaVal
        Xtest(i,:) = fread(fid,[1,784],'uchar'); 
    end
end
yval = ytrain(1:m/n:end);
ytest = yval;

save('MNIST.mat','Xtrain','Xval','Xtest','ytrain','yval','ytest');
end

