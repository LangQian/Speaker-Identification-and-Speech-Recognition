dir_test   = '/Users/lang/Dropbox/CSC2511/a3/speechdata/Testing';
hmm_path = '/Users/lang/Dropbox/CSC2511/a3/A3_ASR/code/hmm81750';
M           = 8;
Q           = 1;

addpath(genpath('./FullBNT-1.0.7'));
total = 0;
correct = 0;
for i=1:30
    disp(i);
    filepath = [dir_test, filesep, 'unkn_', int2str(i), '.phn'];
    mfcc = load([dir_test, filesep, 'unkn_', int2str(i), '.mfcc']);
    num = size(mfcc, 1);
    phoneme_data = textread(filepath, '%s', 'delimiter', '\n');
    total = total + length(phoneme_data);
    hmm  = dir([hmm_path, filesep]);
    hmm = hmm(4:end); % Mac
% hmm = hmm(3:end); % Linux

    for j=1:length(phoneme_data)
        phonems  = strread(phoneme_data{j},'%s','delimiter', ' ');
        start_position = (str2num(phonems{1}) / 128) + 1;
        % end_position = str2num(phonem{2}) / 128;
        end_position = min(str2num(phonems{2}) / 128, num);
        phoneme = phonems{3};
        if strcmp(phoneme, 'h#')
            phoneme = 'sil';
        end
        current = mfcc(start_position:end_position, 1:7);
        top = -Inf;
        flag = '';
        for k=1:length(hmm)
            current_hmm = hmm(k).name;
            model = load([hmm_path, filesep, current_hmm],'-mat');
            model = model.HMM;
            if size(current', 2)==0
                continue;
            end
            likelihood = loglikHMM( model, current');
            if likelihood>top
                top = likelihood; 
                flag = current_hmm;
            end
        end
        if strcmp(flag, phoneme)
            correct = correct + 1;
        end
    end
   
end
percentage = correct/total;
disp(percentage);
