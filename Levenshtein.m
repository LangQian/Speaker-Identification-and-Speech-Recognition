function [SE IE DE LEV_DIST] =Levenshtein(hypothesis,annotation_dir)
% Input:
%	hypothesis: The path to file containing the the recognition hypotheses
%	annotation_dir: The path to directory containing the annotations
%			(Ex. the Testing dir containing all the *.txt files)
% Outputs:
%	SE: proportion of substitution errors over all the hypotheses
%	IE: proportion of insertion errors over all the hypotheses
%	DE: proportion of deletion errors over all the hypotheses
%	LEV_DIST: proportion of overall error in all hypotheses

hyp_data = textread(hypothesis, '%s', 'delimiter', '\n');

total = 0;
SE = 0;
IE = 0;
DE = 0;
LEV_DIST = 0;

for i=1:length(hyp_data)
    hyp = strread(hyp_data{i},'%s','delimiter', ' ');
    reference_data = textread([annotation_dir, filesep, 'unkn_', int2str(i), '.txt'], '%s', 'delimiter', '\n');
    reference = strread(reference_data{1},'%s','delimiter', ' ');
    reference = reference(3:end);
    ref_len = length(reference);
    hyp_len = length(hyp);
    total = total + ref_len;
    R = ones(ref_len+1, hyp_len+1) * Inf;
    for y=1:ref_len+1
        R(y,1)=y-1;
    end
    for z=1:hyp_len+1
        R(1,z)=z-1;
    end
    R(1,1) = 0; 
    backtrack = zeros(ref_len, hyp_len); % store the arrows
    for j=2:(ref_len+1)
        for k=2:(hyp_len+1)
            deletion = R(j-1, k) + 1;
            if strcmp(reference{j - 1}, hyp{k - 1})
                match = R(j-1, k-1);
            else
                match = R(j-1, k-1) + 2; % make sure will not be selected in min function
            end
            substitution = R(j-1, k-1) + 1;
            insertion = R(j, k-1) + 1;
            [R(j, k), index] = min([deletion, match, substitution, insertion]); % use index to indicate arrow
            backtrack(j-1, k-1) = index;
        end
    end
    se = 0;
    ie = 0;
    de = 0;
    m = ref_len;
    n = hyp_len;
    while (m~=0&&n~=0)
        if backtrack(m,n)==1
            de = de + 1;
            m = m -1;
        elseif backtrack(m,n)==3
            se = se + 1;
            m = m - 1;
            n = n - 1;
        elseif backtrack(m,n)==4
            ie = ie + 1;
            n = n - 1;
        else
            m = m - 1;
            n = n - 1;
        end
    end
    
    t = se + ie + de;
    
    disp(['Word error rates for the Hypothesis ', int2str(i)]);
    disp(['SE: ',num2str(se/ref_len)]);
    disp(['IE: ',num2str(ie/ref_len)]);
    disp(['DE: ',num2str(de/ref_len)]);
    disp(['Total: ',num2str(t/ref_len)]);
    
    SE = SE + se;
    IE = IE + ie;
    DE = DE + de;
    
end
SE = SE / total;
IE = IE / total;
DE = DE / total;
LEV_DIST = SE + IE + DE;
disp('Word error rates for the entire data set:');
disp(['SE: ',num2str(SE)]);
disp(['IE: ',num2str(IE)]);
disp(['DE: ',num2str(DE)]);
disp(['Total: ',num2str(LEV_DIST)]);