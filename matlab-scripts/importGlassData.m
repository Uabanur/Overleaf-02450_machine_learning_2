
clear all

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

% standardize dataset
X = (X-mean(X))./std(X,1);

[N, M] = size(X);

%% Set up 2-layer CV partitions

%Outer layer cross-validation: STRATIFIED 
CVout = cvpartition(y,'holdout');


X_par = X(CVout.training(),:); 
y_par = y(CVout.training());
N_par = length(X_par)

X_test = X(CVout.test(),:);
y_test = y(CVout.test());
N_test = length(X_test)

% Inner layer cross-validation: K-FOLD 
K = 10;
CVin = cvpartition(y_par,'Kfold',K);

save('glassDataMatlab')
