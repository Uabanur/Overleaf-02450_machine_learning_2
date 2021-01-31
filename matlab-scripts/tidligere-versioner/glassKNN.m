%%%%%%%%%%%%%%%%%%%% Glass_KNN_v2 %%%%%%%%%%%%%%%%%%%%

%% Import the data
data = xlsread('glass2.xls','glass');

%% Create table
glass2 = table;

%% Allocate imported array to column variable names
glass2.VarName1 = data(:,1);
glass2.VarName2 = data(:,2);
glass2.VarName3 = data(:,3);
glass2.VarName4 = data(:,4);
glass2.VarName5 = data(:,5);
glass2.VarName6 = data(:,6);
glass2.VarName7 = data(:,7);
glass2.VarName8 = data(:,8);
glass2.VarName9 = data(:,9);
glass2.VarName10 = data(:,10);
glass2.VarName11 = data(:,11);

%% Set up X and y
glassdata = table2array(glass2);

y = double(glassdata(:, 11));
X = glassdata(:, 2:10);

numObs = 214;
y = y(1:numObs);
X = X(1:numObs, :);

%normalize dataset
%X = normc(X);
X = (X-mean(X))./std(X,1)

[N, M] = size(X);

%% Set up 2-layer CV partitions

%Outer layer:
CVout = cvpartition(y,'holdout');


X_par = X(CVout.training(),:); 
y_par = y(CVout.training());
N_par = length(X_par)

X_test = X(CVout.test(),:);
y_test = y(CVout.test());
N_test = length(X_test)

% Inner layer:
K = 10;
CVin = cvpartition(y_par,'Kfold',K);

%CVin = cvpartition(N_par,'leaveout')
%K = CVin.NumTestSets;


%% Inner crossvalidation

% K-nearest neighbors parameters
Distance = 'euclidean'; % Distance measure
L = 40; % Maximum number of neighbors

% Variable for classification error
Error = nan(K,L);

for k = 1:K % For each crossvalidation fold
    fprintf('Crossvalidation fold %d/%d\n', k, CVin.NumTestSets);

    % Extract training and validation sets
    X_train = X_par(CVin.training(k), :);
    y_train = y_par(CVin.training(k));
    X_val = X_par(CVin.test(k), :);
    y_val = y_par(CVin.test(k));

    for l = 1:L % For each number of neighbors
        
        % Use knnclassify to find the l nearest neighbors
        knn=fitcknn(X_train, y_train, 'NumNeighbors', l, 'Distance', Distance);
        y_val_est=predict(knn, X_val);
        
        % Compute number of classification errors
        Error(k,l) = sum(y_val~=y_val_est); % Count the number of errors
    end
end

%% Plot the classification error rate
mfig('Error rate');
ValidationErrorSum = sum(Error)./sum(CVin.TestSize)*100;
plot(ValidationErrorSum);
xlabel('Number of neighbors');
ylabel('Classification error rate (%)');

%% Choose best value of K and train on all X_par

% Find lowest validation error and best K
[Evalmin, Kbest] = min(ValidationErrorSum);

% Train KNN with best K on X_par
knnBest=fitcknn(X_par, y_par, 'NumNeighbors', Kbest, 'Distance', Distance);

%predict
y_test_est=predict(knnBest,X_test);

%Compute Generalization Error:
E_gen_est = sum(y_test~=y_test_est); % Count the number of errors
E_gen_est = E_gen_est/N_test



