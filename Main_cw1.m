clear all; close all; 
% Initialisation
init; clc;
[data_train, data_test] = getData('Caltech');

%% Set parameters
param.num = 100;         % Number of trees % intially 50
param.depth = 5;        % Depth of each tree
param.splitNum = 10;     % Number of trials in split function
param.split = 'IG';     % Currently support 'information gain' only

%% Tree Training
trees = growTrees(data_train,param); % this is the final function to be used for tree training

%% Test Random Forest
leaf_assign = testTrees_fast(data_test,trees);

for T = 1:length(trees)
    p_rf(:,:,uint8(T)) = trees(1).prob(leaf_assign(:,uint8(T)),:);
end

% average the results from all trees
p_rf = squeeze(sum(p_rf,3))/length(trees); % Regression
[~,c] = max(p_rf'); % Regression to Classification
accuracy_rf = sum(c==data_test(:,end)')/length(c); % Classification accuracy (for Caltech dataset)
