clear all

%% Load data and CV partitions
load('glassDataMatlabReg')

%% Set up aux vars for ANN, filter X
M = 8; % max 14. if 14, use all 8 weight percent attributes AND 6 type attributes
% if 8, only use all 8 weight attributes.

if (M == 14)
    typesIncluded = 1; % flag for whether types are included
end

if (M == 8)
    typesIncluded = 0; % flag for whether types are included
end

X = X(:, 1:M); % filter based on number of attributes


% Parameters for neural network classifier

% HYPER PARAMETERS FOR ANN REGRESSION
NHiddenUnits = 10; % Number of hidden units to try out
NTrain = 3; % max number of re-trains of neural network


% Variable for classification error
Error = nan(K,length(NHiddenUnits));

% Variables for regression error (continuously overwritten in loop)
Error_train = nan(K,1);
Error_test = nan(K,1);
Error_train_nofeatures = nan(K,1);
Error_test_nofeatures = nan(K,1);
bestnet=cell(K,1);

%% Loop
for k = 1:K % For each crossvalidation fold
    fprintf('Crossvalidation fold %d/%d\n', k, CVin.NumTestSets);

    % Extract training and validation sets
    X_train = X_par(CVin.training(k), :);
    y_train = y_par(CVin.training(k));
    X_val = X_par(CVin.test(k), :);
    y_val = y_par(CVin.test(k));

    % loop over number of hidden units
    for n = 1:NHiddenUnits % 1 through maxHiddenUnits
        
        % loop over number of times to train
        for t = 1:NTrain
            % Fit neural network to training set
            MSEBest = inf;
            netwrk = nr_main(X_train, y_train, X_val, y_val, n);
            if netwrk.mse_train(end)<MSEBest, bestnet{k} = netwrk; MSEBest=netwrk.mse_train(end); MSEBest=netwrk.mse_train(end); end
        end
        
    % Predict model on test and training data    
    y_train_est = bestnet{k}.t_pred_train;    
    y_val_est = bestnet{k}.t_pred_test;        
    
    % Compute least squares error
    Error(k,n) = sum((y_val - y_val_est).^2);
        
    end % end inner n loop
    
end % end outer K loop


%% Plot the classification error rate
sz = 15;
lw = 2;

mfig('Error rate'); clf;
ValidationErrorSum = sum(Error)./sum(CVin.TestSize)*100;
plot(ValidationErrorSum,'b.-', 'MarkerSize', sz, 'LineWidth',lw);
xlabel('Number of hidden units');
ylabel('Classification error rate (%)');

%% Choose best number of hidden units, and train on all X_par

% Find lowest validation error and best number of hidden units
[Evalmin, nBest] = min(ValidationErrorSum);

% Train network on X_par
netBest = nr_main(X_par, y_par, X_test, y_test, nBest);

% Get the predicted output for the test data
y_test_est = netBest.t_pred_test;    

% Compute Generalization Error estimate:
E_gen_est = sum((y_test-y_test_est).^2);
disp('Generalization error estimate: ')
E_gen_est = E_gen_est/N_test

%% For best value of hidden nodes, export error vector
% used for t-test

if (typesIncluded == 1)
    error_for_ttest_ANN_regression_including_type_attributes = Error(:, nBest);
    save('error_for_ttest_ANN_regression_including_type_attributes', 'error_for_ttest_ANN_regression_including_type_attributes');
end

if (typesIncluded == 0)
    error_for_ttest_ANN_regression_excluding_type_attributes = Error(:, nBest);
    save('error_for_ttest_ANN_regression_excluding_type_attributes', 'error_for_ttest_ANN_regression_excluding_type_attributes');
end

%% comparing ANN with just predicting y is mean(y)
y_test_error = abs(y_test_est - y_test  ) ;
y_test_error_simple_predict = abs(y_test - mean(y));

better_than_simple = y_test_error <  y_test_error_simple_predict;
better_than_simple = better_than_simple';

N_succes = sum(better_than_simple);
N_test = length(y_test);
succesRate = N_succes / size(y_test, 1) * 100;
fprintf('ANN prediction was better than predicting y = mean in %.0f/%.0f = %.0f %% cases \n', N_succes, N_test, succesRate);
% histogram ( y_test_error );

%% Display the trained network 
mfig('Trained Network'); clf;
displayNetworkRegression(netBest);

%% Plots
sz = 15;
lw = 2;

mfig('Test: Predicted RI as a function of actual RI value'); clf;
plot(y_test, y_test_est, 'k.', 'MarkerSize', sz, 'LineWidth',lw);
% histogram(y_test_error);

y_test_error_per_std = (y_test_est - y_test ) / std(y);
y_test_error_per_std_simple = (mean(y) - y_test ) / std(y);
% histogram( y_test_error_per_std );


mfig('ANN regression'); clf;  
hold on
plot( y_test*0 + mean(y), 'r.-', 'MarkerSize', sz, 'LineWidth',lw);
plot( y_test_est, 'b.-', 'MarkerSize', sz, 'LineWidth',lw);
plot( y_test, 'k.-', 'MarkerSize', sz, 'LineWidth',lw);


lcn = 'SouthWest';
legend('RI predicted as mean(RI)', 'RI test, predicted using ANN', 'RI test', 'Location', lcn);

xlabel('test index')
ylabel('RI value')


