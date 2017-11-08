dir_train   = '/Users/lang/Dropbox/CSC2511/a3/speechdata/Training';
M           = 5;
Q           = 3;

addpath(genpath('./FullBNT-1.0.7'));

data = dir([dir_train, filesep]);
data = data(4:end); % Mac
% data = data(3:end); % Linux

phonemes_struct = struct();

for i=1:length(data)
    filepath = [dir_train, filesep, data(i).name];
    utterances = dir([filepath, filesep, '*.mfcc']);
   
    for j=1:0.5*length(utterances)
        % read data
        filename = strread(utterances(j).name,'%s','delimiter','.');
        phoneme_data = textread([filepath, filesep, strcat(filename{1},'.phn')], '%s', 'delimiter', '\n');
        mfcc = load([filepath, filesep, utterances(j).name]);
        
        for k=1:length(phoneme_data)
            phonems  = strread(phoneme_data{k},'%s','delimiter', ' ');
            start_position = (str2num(phonems{1}) / 128) + 1;
            % end_position = str2num(phonem{2}) / 128;
            end_position = min(str2num(phonems{2}) / 128, size(mfcc, 1));
            phoneme = phonems{3};
            if strcmp(phoneme, 'h#')
                phoneme = 'sil';
            end
            
            if ~isfield(phonemes_struct, phoneme)
                phonemes_struct.(phoneme) = cell(0);
            end

            phonemes_struct.(phoneme){length(phonemes_struct.(phoneme)) + 1} = mfcc(start_position:end_position, 1:7)';
        end
    end
end

all = fieldnames(phonemes_struct);
for i=1:length(fields(phonemes_struct))
    one = all{i};
    data = phonemes_struct.(one);
    HMM = initHMM(data, M, Q, 'kmeans');
    [HMM, LL] = trainHMM(HMM, data, 15);
    save(one, 'HMM', '-mat');
end