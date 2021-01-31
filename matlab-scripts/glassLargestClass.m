%%% Glass Largest class

%% Load data and variables
load('glassDataMatlab.mat');

%% Inner cross-validation

% Rename number of inner CV-folds:
N_CV = CVin.NumTestSets;


% Variable for classification error
Error_ttest_mode = nan(N_CV,1);

for n = 1:N_CV % For each crossvalidation fold
    fprintf('Crossvalidation fold %d/%d\n', n, CVin.NumTestSets);
    
     % Extract training and validation sets
    y_train = y_par(CVin.training(n));
    y_val = y_par(CVin.test(n));
    
    % Predict that the class is the largest in the training set
    y_val_est = mode(y_train);
    
    % Compute classification error
    Error_ttest_mode(n) = sum(y_val_est ~= y_val);
end

Error_ttest_mode

%% Save no. of classification errors
save LargestClass.mat Error_ttest_mode

    
    