%%%%%%%%%%%%%%%%%%%% GlassDecisionTree %%%%%%%%%%%%%%%%%%%%

%% Import the data
data = xlsread('glass2.xls','glass');

% Attribute names and class names
attributeNames = {'ID','RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','Type'};
classNames = {'Building float', 'Building non-float', ...
              'vehicle float', 'vehicle non-float', ...
              'containers', 'tableware', 'headlamps'}'; %Note the transpose

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

%Defining the glass type as the predicted variable:
y = double(glassdata(:, 11));

%Defining the predictor variables (Excluding ID-tag):
X = glassdata(:, 2:10);
attributeNames = attributeNames(2:10); 

% If you want to exclude points from dataset:
numObs = 214;
y = y(1:numObs);
X = X(1:numObs, :);

% normalize dataset
%X = normc(X);
X = (X-mean(X))./std(X,1);

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

% Pruning levels
prune = 0:10;

% minparent (tree stopping criterion)
minparent = 10;

% Variable for classification error
Error = nan(K,length(prune));

for k = 1:K % For each crossvalidation fold
    fprintf('Crossvalidation fold %d/%d\n', k, CVin.NumTestSets);

    % Extract training and validation sets
    X_train = X_par(CVin.training(k), :);
    y_train = y_par(CVin.training(k));
    X_val = X_par(CVin.test(k), :);
    y_val = y_par(CVin.test(k));

    % Fit classification tree to training set
    T = classregtree(X_train, classNames(y_train), ...
        'method', 'classification', ...
        'splitcriterion', 'gdi', ...
        'categorical', [], ...
        'names', attributeNames, ...
        'prune', 'on', ...
        'minparent', minparent);
    

    % For each pruning level
    for n = 1:length(prune)
        
        % Compute estimates on training and validation splits
        y_val_est = eval(T, X_val, prune(n));
        %y_val_est = str2num(cell2mat(y_val_est)); %Converts to double array
        
        % Compute classification error
        %Error(k,n) = sum(y_val_est ~= y_val);
        Error(k,n) = sum(~strcmp(classNames(y_val),y_val_est));
    end  
end

%% Plot the classification error rate
mfig('Error rate');
ValidationErrorSum = sum(Error)./sum(CVin.TestSize)*100;
plot(ValidationErrorSum);
xlabel('Pruning level');
ylabel('Classification error rate (%)');

%% Choose best pruning level P and train on all X_par

% Find lowest validation error and best pruning level P
[Evalmin, PBest] = min(ValidationErrorSum);

% Train Decision tree on X_par
TBest = classregtree(X_par, classNames(y_par), ...
        'method', 'classification', ...
        'splitcriterion', 'gdi', ...
        'categorical', [], ...
        'prune', 'on', ...
        'minparent', minparent);

% Visualise Decision tree:
view(TBest)

% Predict class
y_test_est = eval(TBest,X_test,PBest);
%y_test_est = str2num(cell2mat(y_test_est)); %Converts to double array

%Compute Generalization Error estimate:
E_gen_est = sum(~strcmp(classNames(y_test),y_test_est)); % Count the number of errors
E_gen_est = E_gen_est/N_test



