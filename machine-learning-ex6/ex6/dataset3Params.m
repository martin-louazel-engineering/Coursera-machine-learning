function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

errVec = zeros(8, 8);
ind_i = 1;
ind_j = 1;
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
    errVec(ind_i, ind_j) = error;
    
    ind_j += 1;
    if ind_j > 8
      ind_j = 1;
    endif
  endfor
  ind_i += 1;
endfor

i = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
j = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
[ind_i, ind_j] = find(errVec == min(min(errVec)) );
C = i(ind_i);
sigma = j(ind_j);





% =========================================================================

end
