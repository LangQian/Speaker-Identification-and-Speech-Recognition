% function gmmClassify( dir_train, dir_test, max_iter, epsilon, M )

dir_train = '';
dir_test = '';
max_iter = 1000;
epsilon = 0.3;
M = 8;
% calculate gmms
gmms = gmmTrain( dir_train, max_iter, epsilon, M );
result = cell(1, 30, length(gmms));
% calculate likelihoods
for i=1:30
    path = ['unkn_', int2str(i), '.mfcc'];
    data = load([dir_test, filesep, path]);
    num = size(data, 1);
    dim = size(data, 2);
    for j=1:length(gmms)
        result{i}{j}.name = gmms{j}.name;
        % calculate the observation probability for mixture components
        b = zeros(num, M);
        for k=1:M
            means = gmms{j}.means(:, k);
            cov = gmms{j}.cov(:, :, k);
            diff = data - repmat(means.', num, 1); % meet the dimensions
            diag_cov = diag(cov);
            upper = (diff .* diff) ./ repmat(diag_cov.', num, 1);
            lower = -0.5 * log(2 * pi * repmat(diag_cov.', num, 1));
            val = -0.5 * upper + lower;
            b(:, k) = sum(val, 2);
        end
        b = exp(b); % num x M
        
        % calculate likelihood
        likelihood = sum(log(b * gmms{j}.weights'), 1);
        result{i}{j}.likelihood = likelihood;
    end
end

for i=1:30
    index = [1,2,3,4,5];
    value = [result{i}{1}.likelihood, result{i}{2}.likelihood, result{i}{3}.likelihood, result{i}{4}.likelihood, result{i}{5}.likelihood];
    [value, ind] = sort(value, 'descend');
    index = index(ind);
    for j=6:length(gmms)
        [val, flag] = min(value);
        if result{i}{j}.likelihood > val
            index(flag) = j;
            value(flag) = result{i}{j}.likelihood;
            [value, ind] = sort(value, 'descend');
            index = index(ind);
        end
    end
    disp(['Test Utterance ', int2str(i), ':'])
    for k=1:5
        disp(result{i}{index(k)}.name);
        disp(result{i}{index(k)}.likelihood);
    end
end
    
% end
