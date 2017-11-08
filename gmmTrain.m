function gmms = gmmTrain( dir_train, max_iter, epsilon, M )
% gmmTain
%
%  inputs:  dir_train  : a string pointing to the high-level
%                        directory containing each speaker directory
%           max_iter   : maximum number of training iterations (integer)
%           epsilon    : minimum improvement for iteration (float)
%           M          : number of Gaussians/mixture (integer)
%
%  output:  gmms       : a 1xN cell array. The i^th element is a structure
%                        with this structure:
%                            gmm.name    : string - the name of the speaker
%                            gmm.weights : 1xM vector of GMM weights
%                            gmm.means   : DxM matrix of means (each column 
%                                          is a vector
%                            gmm.cov     : DxDxM matrix of covariances. 
%                                          (:,:,i) is for i^th mixture

data = dir([dir_train, filesep]);
data = data(4:end); % Mac
% data = data(3:end); % Linux

% initialization of gmms
gmms = cell(1, length(data));
for i=1:length(data)
    gmms{i}.name = data(i).name;
end

% calculate the rest parts
for i=1:length(data)
    folder_name = [dir_train, filesep, gmms{i}.name, filesep];
    parameters = calculate_gmm(folder_name, max_iter, epsilon, M);
    gmms{i}.weights = parameters.weights;
    gmms{i}.means = parameters.means;
    gmms{i}.cov = parameters.cov;
end

end

function parameters = calculate_gmm(path, max_iter, epsilon, M)
    % load data
    datafiles = dir([path, '*.mfcc']);
%     data = {};
%     for i=1:length(datafiles)
%         data_path = [path, filesep, datafiles(i).name];
%         data = [data, load(data_path)];
%     end
    
    data = load([path, filesep, datafiles(1).name]);
    for j=2:length(datafiles)
        datafile = datafiles(j).name;
        nextData = load([path, filesep, datafile]);
        data = [data; nextData];
    end

    num = size(data, 1);
    dim = size(data, 2);
    % initialization
    parameters.weights(1,1:M) = 1 / M;
    % Initialize each mu to a random vector from the data
    parameters.means = data(ceil(rand(1, M) * num), :);
    parameters.means = parameters.means';
    parameters.cov = zeros(dim, dim, M);
    for i=1:M
        parameters.cov(:, :, i) = eye(dim, dim);
    end
    
    % initializtion of EM algorithm
    iteration = 0;
    prev_L = -Inf ;
    improvement = Inf;
    
    % EM algorithm
    while iteration <= max_iter && improvement >= epsilon
        [parameters, likelihood] = em(data, parameters, num, dim, M);
        improvement = likelihood - prev_L;
        prev_L = likelihood;
        iteration = iteration + 1;
    end
end

function [parameters, likelihood] = em(data, parameters, num, dim, M)
    % calculate the observation probability for mixture components
    b = zeros(num, M);
    for i=1:M
        means = parameters.means(:, i);
        diff = data - repmat(means.', num, 1); % meet the dimensions
        diag_cov = diag(parameters.cov(:, :, i));
        upper = (diff .* diff) ./ repmat(diag_cov.', num, 1);
        lower = -0.5 * log(2 * pi * repmat(diag_cov.', num, 1));
        val = -0.5 * upper + lower;
        b(:, i) = sum(val, 2);
    end
    b = exp(b); % num x M
    
    % calculate the prior probability of the Gaussian components
    upper = repmat(parameters.weights, num, 1) .* b;
    lower = repmat(b * parameters.weights', 1, M);
    pp = upper ./ lower; % num * M
    
    % calculate likelihood
    likelihood = sum(log(b * parameters.weights'), 1);
    
    % update weights
    parameters.weights = sum(pp, 1) ./ num;
    
    % update means
    parameters.means = data' * pp ./ repmat(sum(pp, 1), dim, 1);
    
    % update covariances
    square_data = data .* data;
    upper = square_data' * pp;
    left = upper ./ repmat(sum(pp, 1), dim, 1);
    squared_means = parameters.means .* parameters.means;
    variance = left - squared_means;
    for i=1:M
        parameters.cov(:, :, i) = diag(variance(:, 1));
    end
end







