%% Clear Workspace and close all files, to run the new test %%
clear all; close all; clc;
load avila.dat
%% Split the data according to the possible Outputs %%
OutputNumber = max(avila(:,end));
Class_Data = cell(OutputNumber,1);
for count = 1:OutputNumber
    Class_Data{count,1} = avila( avila(:,end) == count, :);
end
%% Seperate dataset to D_trn, D_val, D_chk %%
%~ 1st 60% is the Data for Training, next 20% is the Data for Validating and last 20% is the Data for Chechking. ~%
TRN_persent = 0.6; VAL_persent = 0.2; CHK_persent = 0.2;
D_trn = []; D_val = []; D_chk = [];
for count = 1:OutputNumber
    currData = Class_Data{count,1};
    [rows, ~] = size(currData); rng(1);
    currData = currData(randperm(size(currData,1)),:);
    D_trn = [D_trn; currData(1:round(TRN_persent * rows),:)];
    D_val = [D_val; currData(round(TRN_persent * rows)+1:round(TRN_persent * rows) + 1 + round(VAL_persent * rows),:)];
    D_chk = [D_chk; currData(round(TRN_persent * rows)+round(VAL_persent * rows)+2:end,:)];
end
D_trn = D_trn(randperm(size(D_trn,1)),:);
D_val = D_val(randperm(size(D_val,1)),:);
D_chk = D_chk(randperm(size(D_chk,1)),:);
%% Create the Model and Run Test%%
% NR  = [4  5   7    9   11];
RADII = [1 0.8 0.71 0.6 0.4];
%~ Create the cell arrays to hold the results of the training ~%
Models = cell(length(RADII), 1); TrainedModels = cell(length(RADII), 1);
TrainingError = cell(length(RADII), 1); CheckingError = cell(length(RADII), 1);

options = genfisOptions('SubtractiveClustering');
options.SquashFactor = 0.5;
for count = 1:length(RADII)
    options.ClusterInfluenceRange = RADII(count);
    Init_Model = genfis(D_trn(:,1:end-1), D_trn(:,end), options);
    for i = 1:length(Init_Model.output(1).mf)
        Init_Model.output(1).mf(i).type = 'constant';
         Init_Model.output(1).mf(i).params = 6.5;
    end
    NR = length(Init_Model.Rules());
    fprintf(' For radii = %.2f \t NR = %d.\n', RADII(count), NR);
    [Models{count}, TrainingError{count}, ~, TrainedModels{count}, CheckingError{count}] = anfis(D_trn, Init_Model, ...
                                                                                             200, [0 0 0 0], D_val);
    NaNs =  ismissing(CheckingError{count});
    if max(NaNs) 
        fprintf('\t Oooops NaN value was found @ NR = %d.\n',NR);
    end
end
%% Evaluate the model %%
Model_Output = cell(length(RADII), 1);
for count = 1:length(RADII)
    Model_Output{count} = round(evalfis(TrainedModels{count}, D_chk(:,1:end-1)));
    Model_Output{count}(Model_Output{count} < 1) = 1;
    Model_Output{count}(Model_Output{count} > OutputNumber) = OutputNumber;
end
%% Compute the Metrics %%
%~ Create Error Matrix ~%
ErrorMatrix = cell(length(RADII), 1);
RealOutputData = D_chk(:,end);
for count = 1:length(RADII)
    PredictedOutputData = Model_Output{count};
    for i = 1:OutputNumber          % Real Value
        for j = 1:OutputNumber      % Predicted
            ErrorMatrix{count}(i, j) = sum((RealOutputData == i) .* (PredictedOutputData == j));
        end
    end
end
%~ Overall Accuracy ~%
OverallAccuracy = zeros(length(RADII), 1);
for count = 1:length(RADII)
    OverallAccuracy(count) = trace(ErrorMatrix{count}) / length(D_chk(:,end));
end
%~ Producer's and Users's accuracy ~%
x_ir = cell(length(RADII), 1); x_jc = cell(length(RADII), 1);
PA = cell(length(RADII), 1);    UA = cell(length(RADII), 1);
for count = 1:length(RADII)
    x_ir{count} = sum( ErrorMatrix{count}, 2);      %Sum of each row
    x_jc{count} = sum( ErrorMatrix{count}, 1);      %Sum of each column
end
for count = 1:length(RADII)
    for i = 1:OutputNumber
        PA{count}(i) = ErrorMatrix{count}(i,i) / x_jc{count}(i);
        UA{count}(i) = ErrorMatrix{count}(i,i) / x_ir{count}(i);
    end
end
%~ Compute Khat ~%
Khat = zeros(length(RADII), 1);
N = length(D_chk(:,end));
for count = 1:length(RADII)
    Khat(count) = (N * trace(ErrorMatrix{count}) - sum(x_ir{count} .* x_jc{count})) / (N^2 - sum(x_ir{count} .* x_jc{count}));
end
%% Plot Metrics %%
%~ Membership Functions ~%
for count = 1:length(RADII)
    figure();
    h = zeros(5,1);
    for i = 1:length(TrainedModels{count}.input)/2
        h(i) = subplot(3,2,i);
        [x, mf] = plotmf(TrainedModels{count}, 'input', i);
        plot(x, mf);
        xlabel(['Input ', num2str(i)]); ylabel(' Degree of Membership ' );
    end
    h(5) = subplot(3,2,5); % the last (odd) axes
    pos = get(h,'Position');
    new = mean(cellfun(@(v)v(1),pos(1:2)));
    set(h(5),'Position',[new,pos{end}(2:end)])
    sgtitle(['Model ', num2str(count), ': First 5 Inputs']);
%~ Second Half ~%
    figure();
    start = length(TrainedModels{count}.input)/2;
    for i = start+1:length(TrainedModels{count}.input)
        h(i-start) = subplot(3,2,i-start);
        [x, mf] = plotmf(TrainedModels{count}, 'input', i);
        plot(x, mf);
        xlabel(['Input ', num2str(i)]); ylabel(' Degree of Membership ' );
    end
    h(5) = subplot(3,2,5); % the last (odd) axes
    pos = get(h,'Position');
    new = mean(cellfun(@(v)v(1),pos(1:2)));
    set(h(5),'Position',[new,pos{end}(2:end)])
    sgtitle(['Model ', num2str(count), ': Last 5 Inputs']);
end
%~ Learning Curves %%
for count = 1:length(RADII)
    figure()
    sgtitle(['Learning Curve of Model ', num2str(count)])
    plot(TrainingError{count}); hold on;
    plot(CheckingError{count})
    xlabel('Epochs'); ylabel('Error');
    xlim([0 200]); legend('Training Error', 'Checking Error');
end
%~ Confusion Matrix %%
for count = 1:length(RADII)
    figure()
    sgtitle(['Error Matrix of Model ', num2str(count)])
    confusionchart(D_chk(:,end),Model_Output{count});
end
