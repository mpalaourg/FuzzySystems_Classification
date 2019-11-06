%% Clear Workspace and close all files, to run the new test %%
clear all; close all; clc;
load Isolet.dat
%% Run relieff to compute the Main Features%%
ranks = relieff(Isolet(:,1:end-1),Isolet(:,end), 200, 'method','classification');
%% Split the data according to the possible Outputs %%
OutputNumber = max(Isolet(:,end));
Class_Data = cell(OutputNumber,1);
for count = 1:OutputNumber
    Class_Data{count,1} = Isolet( Isolet(:,end) == count, :);
end
%% Seperate dataset to D_trn, D_val, D_chk %%
%~ 1st 60% is the Data for Training, next 20% is the Data for Validating and last 20% is the Data for Chechking. ~%
TRN_persent = 0.60; VAL_persent = 0.2; CHK_persent = 0.2;
D_trn = []; D_val = []; D_chk = [];
for count = 1:OutputNumber
    currData = Class_Data{count,1};
    [rows, ~] = size(currData);
    currData = currData(randperm(size(currData,1)),:);
    D_trn = [D_trn; currData(1:round(TRN_persent * rows),:)]; 
    D_val = [D_val; currData(round(TRN_persent * rows)+1:round(TRN_persent * rows) + 1 + round(VAL_persent * rows),:)];
    D_chk = [D_chk; currData(round(TRN_persent * rows)+round(VAL_persent * rows)+2:end,:)];
end
D_trn = D_trn(randperm(size(D_trn,1)),:);
D_val = D_val(randperm(size(D_val,1)),:);
D_chk = D_chk(randperm(size(D_chk,1)),:);
%% Create the model Only with the Wanted Features %%
NF = 21; NR = 20;
opt = NaN(4,1);
opt(4) = 0;
fprintf('Initialize the Model.\n');
Init_Model = genfis3(D_trn(:,ranks(1:NF)), D_trn(:,end), 'sugeno', NR, opt);
for i = 1:length(Init_Model.output(1).mf)
    Init_Model.output(1).mf(i).type = 'constant';
    Init_Model.output(1).mf(i).params = 13.5;
end
%% Train the Model with anfis %%
Options = anfisOptions('InitialFIS', Init_Model, 'ValidationData', [D_val(:,ranks(1:NF)) D_val(:,end) ] , 'EpochNumber', 250);
Options.DisplayANFISInformation = 0;
Options.DisplayErrorValues = 0;
Options.DisplayStepSize = 0;
Options.DisplayFinalResults = 0;
fprintf('Starting tuning the Model...\n');
[TRN_FIS, TRN_Error, ~, CHK_FIS, CHK_Error] = anfis([D_trn(:,ranks(1:NF)) D_trn(:,end)], Options);
fprintf('Finish tuning the Model.\n');
%% Evaluate the model %%
Model_Output = round(evalfis(CHK_FIS, D_chk(:,ranks(1:NF))));
Model_Output(Model_Output < 1) = 1;
Model_Output(Model_Output > OutputNumber) = OutputNumber;
%% Compute the Metrics %%
%~ Create Error Matrix ~%
ErrorMatrix = zeros(OutputNumber, OutputNumber);
RealOutputData = D_chk(:,end);
PredictedOutputData = Model_Output;
for i = 1:OutputNumber          % Real Value
    for j = 1:OutputNumber      % Predicted
        ErrorMatrix(i, j) = sum((RealOutputData == i) .* (PredictedOutputData == j));
    end
end
%~ Overall Accuracy ~%
OverallAccuracy = trace(ErrorMatrix) / length(D_chk(:,end));
%~ Producer's and Users's accuracy ~%
PA = zeros(OutputNumber, 1); UA = zeros(OutputNumber, 1);
x_ir = sum( ErrorMatrix, 2);      %Sum of each row
x_jc = sum( ErrorMatrix, 1);      %Sum of each column
for i = 1:OutputNumber
    PA(i) = ErrorMatrix(i,i) / x_jc(i);
    UA(i) = ErrorMatrix(i,i) / x_ir(i);
end
%~ Compute Khat ~%
N = length(D_chk(:,end));
Khat = (N * trace(ErrorMatrix) - sum(x_ir .* x_jc)) / (N^2 - sum(x_ir .* x_jc));
%% Plot Some Results %%
%~ Membership Functions of Initial Model ~%
figure('Name','Some Membership Functions','NumberTitle','off')
sgtitle('Some Membership Functions')
subplot(2,2,1);
[xOut,yOut] = plotmf(Init_Model,'input',2);
[xOut2,yOut2] = plotmf(CHK_FIS,'input',2);
plot(xOut(:,3),yOut(:,3)); hold on
plot(xOut2(:,3),yOut2(:,3));
xlabel('MF3 @ Input 2'); ylabel('Degree of Membership');
legend('Initial Model', 'Tuned Model');
subplot(2,2,2);
[xOut,yOut] = plotmf(Init_Model,'input',3);
[xOut2,yOut2] = plotmf(CHK_FIS,'input',3);
plot(xOut(:,12),yOut(:,12)); hold on
plot(xOut2(:,12),yOut2(:,12));
xlabel('MF12 @ Input 3'); ylabel('Degree of Membership');
legend('Initial Model', 'Tuned Model');
subplot(2,2,3);
[xOut,yOut] = plotmf(Init_Model,'input',5);
[xOut2,yOut2] = plotmf(CHK_FIS,'input',5);
plot(xOut(:,3),yOut(:,3)); hold on
plot(xOut2(:,3),yOut2(:,3));
xlabel('MF3 @ Input 5'); ylabel('Degree of Membership');
legend('Initial Model', 'Tuned Model');
subplot(2,2,4);
[xOut,yOut] = plotmf(Init_Model,'input',6);
[xOut2,yOut2] = plotmf(CHK_FIS,'input',6);
plot(xOut(:,1),yOut(:,1)); hold on
plot(xOut2(:,1),yOut2(:,1));
xlabel('MF1 @ Input 6'); ylabel('Degree of Membership');
legend('Initial Model', 'Tuned Model');
%~ Predictions ~%
figure('Name','Predictions for Optimal Features','NumberTitle','off')
sgtitle(' Prediction and Real Values for Optimal Features' )
plot(D_chk(1:100,end),'bo')
hold on;
plot(Model_Output(1:100),'rx')
xlabel('Index'); ylabel('Output');
xlim([0 length(Model_Output(1:100))]);
legend('Real Values', 'Predictions');
%~ Learning Curves ~%
figure('Name','Training and Checking Error for Optimal Features','NumberTitle','off')
sgtitle(' Learning Curve of TSK Model for Optimal Features' )
plot(TRN_Error)
hold on;
plot(CHK_Error)
xlabel('Epochs'); ylabel('Error');
xlim([0 250]);
legend('Training Error', 'Checking Error');
%~ Confusion Matrix ~%
figure()
sgtitle('Error Matrix of Optimal Model ')
confusionchart(D_chk(:,end),Model_Output,'Normalization','row-normalized','RowSummary','row-normalized');
