
clear all

%% Load data and CV partitions
load('glassDataMatlab')

%% Set up aux vars for ANN, filter X
C = 7; % predict classes 1 through 7
M = 9; % max 9. if 9, use RI and all weight percent attributes
X = X(:, 1:M); % filter based on number of attributes


% Number of hidden units to try out
maxHiddenUnits = 10;
NHiddenUnits = 1 : maxHiddenUnits;

% Variable for classification error
Error = nan(K,length(NHiddenUnits));


for k = 1:K % For each crossvalidation fold
    fprintf('Crossvalidation fold %d/%d\n', k, CVin.NumTestSets);

    % Extract training and validation sets
    X_train = X_par(CVin.training(k), :);
    y_train = y_par(CVin.training(k));
    X_val = X_par(CVin.test(k), :);
    y_val = y_par(CVin.test(k));

    %% Fit multiclass neural network to training set
    % for increasing num of hidden units
    
    for n = NHiddenUnits % 1 through maxHiddenUnits
    net = nc_main(X_train, y_train, X_val, y_val, n);
   
    %% Compute results on test data
    % Get the predicted output for the test data
    Y_val_est = nc_eval(net, X_val);    
   
    % Compute the class index by finding the class with highest probability from the neural
    % network
    y_val_est = max_idx(Y_val_est);

    % Compute least squares error
    Error(k,n) = sum(y_val ~= y_val_est);
    end
    
end

%% Plot the classification error rate
sz = 15;
lw = 2;

mfig('Error rate');
ValidationErrorSum = sum(Error)./sum(CVin.TestSize)*100;
plot(ValidationErrorSum, 'MarkerSize', sz, 'LineWidth',lw);
xlabel('Number of hidden units');
ylabel('Classification error rate (%)');

%% Choose best number of hidden units, and train on all X_par

% Find lowest validation error and best number of hidden units
[Evalmin, nBest] = min(ValidationErrorSum);

% Train network on X_par
netBest = nc_main(X_par, y_par, X_test, y_test, nBest);

% Get the predicted output for the test data
Y_test_est = nc_eval(net, X_test);    
   
 % Compute the class index by finding the class with highest probability from the neural
 % network
 y_test_est = max_idx(Y_test_est);

%Compute Generalization Error estimate:
E_gen_est = sum(y_test ~= y_test_est); % Count the number of errors
disp('Generalization error estimate: ')
E_gen_est = E_gen_est/N_test

%% For best value of hidden nodes, export error vector
% used for t-test

error_for_ttest_ANN_classification = Error(:, nBest);
error_rate_in_percent_for_ttest_ANN_classification = error_for_ttest_ANN_classification/N_test*100;
save('error_for_ttest_ANN_classification', 'error_for_ttest_ANN_classification');
save('error_rate_in_percent_for_ttest_ANN_classification', 'error_rate_in_percent_for_ttest_ANN_classification');


%% Plot results
% Display trained network
mfig('Trained network'); clf;   
displayNetworkClassification(netBest)
 
% Display decision boundaries
mfig('Decision Boundaries'); clf;   
dbplot(X_test, y_test, @(X) max_idx(nc_eval(net, X)));
xlabel('PCA component 1');
ylabel('PCA component 2');

mfig('Histogram: difference between test class and test class predicted'); clf; 
hist(y_test - y_test_est)
% interpretation: the most common is to predict correctly.
% second-most common, to predict the class off-by-one, 
% e.g. predict 4 when the real class is 3.
% so class numbers are not independent!

sz = 15;
lw = 2;
y_test_est_prob =  max(Y_test_est,[],2);

disp('Avg prob of y_test_est:')
mean(y_test_est_prob)

mfig('Probability of class prediction'); clf;  
plot(y_test_est_prob, 'r.-', 'MarkerSize', sz, 'LineWidth',lw)

mfig('Probability of class prediction: boxplot'); clf;  
boxplot(y_test_est_prob)
% interpretation: most predictions are far from certain
% many guesses are in the 50 % - 70 % range

plot(y_test, y_test_est_prob ,'r.', 'MarkerSize', sz, 'LineWidth',lw)
% interpretation: class 7 is easy to guess.
% class 1 and 2 are difficult (probs are low)

plot(y_test_est - y_test, y_test_est_prob ,'r.', 'MarkerSize', sz, 'LineWidth',lw)

% plan for new figure:
% x-axis: class 1 through 7 (class of point in y_test)
% y-axis: prob of prediction on current test point
% colors in plot: BLUE DOT means correct prediction, RED DOT means incorrect
