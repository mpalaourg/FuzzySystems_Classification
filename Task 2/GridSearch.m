%% Clear Workspace and close all files, to run the new test %%
clear all; close all; clc;
load Isolet.dat
%% Random Permutate the array, to shuffle the rows and run relieff to compute the Main Features%%
ranks = relieff(Isolet(:,1:end-1),Isolet(:,end), 200, 'method','classification');
%% Pre-processing of data %%
%~ Check for NaN ~%
NaNs =  ismissing(Isolet(:));
if ~max(NaNs), fprintf('None NaN value was found in dataset.\n'); end
%~ Check for Infs ~%
Infs =  isinf(Isolet(:));
if ~max(Infs), fprintf('None Inf value was found in dataset.\n'); end
%% Split the data according to the possible Outputs %%
OutputNumber = max(Isolet(:,end));
Class_Data = cell(OutputNumber,1);
for count = 1:OutputNumber
    Class_Data{count,1} = Isolet( Isolet(:,end) == count, :);
end
%% Seperate dataset to D_trn, D_val, D_chk %%
%~ 1st 60% is the Data for Training and must be done for all Classes. ~%
TRN_persent = 0.6; VAL_persent = 0.2; CHK_persent = 0.2;
D_trn = cell(OutputNumber, 1); Model_Trn_Data = []; Model_Val_Data = [];
for count = 1:OutputNumber
    currData = Class_Data{count,1};
    [rows, ~] = size(currData);
    D_trn{count} = currData(1:round(TRN_persent * rows),:);
    Model_Trn_Data = [Model_Trn_Data; D_trn{count}];
    Model_Val_Data = [Model_Val_Data; currData(round(TRN_persent * rows)+1:round(TRN_persent * rows) + 1 + round(VAL_persent * rows),:)];
end
Model_Trn_Data = Model_Trn_Data(randperm(size(Model_Trn_Data,1)),:);
Model_Val_Data = Model_Val_Data(randperm(size(Model_Val_Data,1)),:);
%% Create Cross Validation Folds and Run the Tests%%
%~ For each value of NF and NR a model will be initialized by genfis3 and
%~ the Training Data will be seperated to 5 folds {80% Training - 20% Validating}
%~ by cvpartition. The difference with Work_3, is that cv partition must be
%~ executed for each and every one of the classes separately and then combine
%~ those partitions to get the Training Data evenly seperated for each class.
%~ Each fold will be used to tuned anfis, for 100 Epochs and the Overall
%~ Accuracy of the Epochs will be saved as the Fold Overall Accuracy. Finally,
%~ the Mean of this Overall Accuracy will be Computed as the metric to find the Optimal Model.
fileID = fopen('ConsoleLog_Final_Test.txt', 'a');
NF = [3 9 16 21]; 
NR = [4 8 12 16 20];
c = cell(OutputNumber, 1); OverallAccuracy = zeros(length(NF), length(NR));
opt = NaN(4,1); opt(4) = 0;
for f = 1:length(NF)
    fprintf('Testing for # of Features: %d\n', NF(f));
    fprintf(fileID,'Testing for # of Features: %d\n', NF(f));
    for r = 1:length(NR)
        t_Start = tic;
        fprintf('\tTesting for # of Rules: %d', NR(r));
        fprintf(fileID,'\tTesting for # of Rules: %d', NR(r));
        Init_Model = genfis3(Model_Trn_Data(:,ranks(1:NF(f))), Model_Trn_Data(:,end), 'sugeno', NR(r), opt);
        for i = 1:length(Init_Model.output(1).mf)
            Init_Model.output(1).mf(i).type = 'constant';
            Init_Model.output(1).mf(i).params = 13.5;
        end
        for count = 1:OutputNumber
            currTrnData = D_trn{count,1};
            c{count} = cvpartition(currTrnData(:,end), 'Kfold', 5);
        end
        Curr_OA = 0;
        for k = 1:5
            Train_Set_In    = []; Train_Set_Out    = [];
            Validate_Set_In = []; Validate_Set_Out = [];
            for count = 1:OutputNumber
                currCvPart = c{count};
                currTrnData = D_trn{count,1};
                training_idx = currCvPart.training(k);
                testing_idx  = currCvPart.test(k);
                
                Train_Set_In     = [Train_Set_In;     currTrnData(training_idx, ranks(1:NF(f)))];
                Train_Set_Out    = [Train_Set_Out;    currTrnData(training_idx, end)];
                Validate_Set_In  = [Validate_Set_In;  currTrnData(testing_idx, ranks(1:NF(f)))];
                Validate_Set_Out = [Validate_Set_Out; currTrnData(testing_idx, end)];
            end
            Train_Set = [Train_Set_In Train_Set_Out];
            Train_Set = Train_Set(randperm(size(Train_Set,1)),:);
            Validate_Set = [Validate_Set_In Validate_Set_Out];
            Validate_Set = Validate_Set(randperm(size(Validate_Set,1)),:);
            Options = anfisOptions('InitialFIS', Init_Model, 'ValidationData', Validate_Set, 'EpochNumber', 100);
            Options.DisplayANFISInformation = 0;
            Options.DisplayErrorValues = 0;
            Options.DisplayStepSize = 0;
            Options.DisplayFinalResults = 0;
            
            [~, TRN_Error, ~, CHK_FIS, CHK_Error] = anfis(Train_Set, Options);
            NaNs =  ismissing(CHK_Error(:));
%~ If a NaN value will be returned by anfis, then set the Fold Overall Accuracy to Inf and continue to the next NR(r) ~%
            if max(NaNs)
                fprintf('\t Oooops NaN value was found @ NF = %d, NR = %d.', NF(f), NR(r));
                fprintf(fileID,'Oooops NaN value was found @ NF = %d, NR = %d.\n', NF(f), NR(r));
                Epoch_OA = Inf;
                Curr_OA = Curr_OA + Epoch_OA;
                break;
            else
                Model_Output = round(evalfis(Model_Val_Data(:, ranks(1:NF(f))), CHK_FIS));
                ErrorMatrix = zeros(length(OutputNumber), 1);
                RealOutputData = Model_Val_Data(:,end);
                PredictedOutputData = Model_Output;
                for i = 1:OutputNumber          % Real Value
                    for j = 1:OutputNumber      % Predicted
                        ErrorMatrix(i, j) = sum((RealOutputData == i) .* (PredictedOutputData == j));
                    end
                end
                %~ Overall Accuracy ~%
                OA = trace(ErrorMatrix) / length(Model_Val_Data(:,end));
                Curr_OA   = Curr_OA + OA;
            end
        end
        t_End = toc(t_Start);
        fprintf('\t Elapsed Time: %f sec.\n', t_End);
        fprintf(fileID,'\t Elapsed Time: %f sec.\n', t_End);
        OverallAccuracy(f, r) = Curr_OA / 5;
    end
    save Final_Test.mat
end
fclose(fileID);
%% Plots The Overall Accuracy of the Grid Search at 2-D Bar and 3D %%
Optimal_Value = max(max(OverallAccuracy));
[Opt_x,Opt_y] = find(OverallAccuracy == Optimal_Value);
fprintf('The Optimal Model is for NF = %d, NR = %d.\n', NF(Opt_x), NR(Opt_y));
%~ Bar Plot 2D ~%
figure('Name','Bar Plot 2D','NumberTitle','off')
sgtitle('Bar Plot Overall Accuracy - 2D')
subplot(2,2,1);
bar(OverallAccuracy(1,:))
xlabel('Number of Rules');
ylabel('Mean Overall Accuracy');
xticklabels(string(NR));
legend([num2str(NF(1)),' features'])
subplot(2,2,2);
bar(OverallAccuracy(2,:))
xlabel('Number of Rules');
ylabel('Mean Overall Accuracy');
xticklabels(string(NR));
legend([num2str(NF(2)),' features'])

subplot(2,2,3);
bar(OverallAccuracy(3,:))
xlabel('Number of Rules');
ylabel('Mean Overall Accuracy');
xticklabels(string(NR));
legend([num2str(NF(3)),' features'])

subplot(2,2,4);
bar(OverallAccuracy(4,:))
xlabel('Number of Rules');
ylabel('Mean Overall Accuracy');
xticklabels(string(NR));
legend([num2str(NF(4)),' features'])
%~ Bar Plot 3D ~%
figure('Name','Bar Plot 3D','NumberTitle','off')
b = bar3(OverallAccuracy);
for k = 1:length(b)
    zdata = b(k).ZData;
    b(k).CData = zdata;
    b(k).FaceColor = 'interp';
end
xlabel('Number of Rules');
ylabel('Number of Features');
zlabel('Mean Overall Accuracy');
xticklabels(string(NR));
yticklabels(string(NF));
