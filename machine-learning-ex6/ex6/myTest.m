load('ex6data3.mat');

errVec = zeros(64, 1);
ind = 1;
for i = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
  for j = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    % SVM Parameters
    C = i;
    sigma = j;

    % We set the tolerance and max_passes lower here so that the code will run
    % faster. However, in practice, you will want to run the training to
    % convergence.
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
    pred = svmPredict(model, Xval);

    error = mean(double(pred ~= yval));
    errVec(ind) = error;
    ind += 1;
  endfor
endfor
errVec