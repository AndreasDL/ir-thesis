function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
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

values = [0.01 0.03 0.1 0.3 1 3 10 30];
%errors = zeros(8,8);

minval = 99999999999999999999999999999999999999;
minCdex = -1;
minSdex = -1;

for cindex=1:8
	for sindex=1:8
		C = values(cindex);
		sigma = values(sindex);
		%train on test
		% We set the tolerance and max_passes lower here so that the code will run
		% faster. However, in practice, you will want to run the training to
		% convergence.
		model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
		%visualizeBoundary(X, y, model);

		%value of cv?
		pred = svmPredict(model, Xval);
		err = mean(double(pred ~= yval));
		%errors(cindex,sindex) = mean(double(pred ~= y));
		if (err < minval)
			minCdex = cindex;
			minSdex = sindex;
			minval = err;
		end
	end
end

%model= svmTrain(X, y, 0.01, @(x1, x2) gaussianKernel(x1, x2, 0.01));
%visualizeBoundary(X, y, model);
%pause;

C = values(minCdex)
sigma = values(minSdex)


% =========================================================================

end