function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

for j = 1:size(X,1)

	minCost = 0;
	for h=1:size(X,2) %for all dimensions
		minCost += (X(j,h) - centroids(1,h))**2;
	end
	minCentroidIndex = 1;

	for i=2:K
		cost = 0;
		for h=1:size(X,2) %for all dimensions
			cost += (X(j,h) - centroids(i,h))**2;
		end
	
		if (cost < minCost)
			minCost = cost;
			minCentroidIndex = i;
		end
	end

	idx(j) = minCentroidIndex;
end






% =============================================================

end

