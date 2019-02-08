clear all; close all; 
% Initialisation
init; 
clc;


%% Train trees from descriptors and corresponding labels
[ data_Train, data_Test ] = getData_Two_Modes('RF_Codebook'); % or 'KMEAN_Codebook'

%% Test Random Forest (Classification)
% Set parameters
param.num = 50;         % Number of trees % intially 100
param.depth = 10;        % Depth of each tree
param.splitNum = 40;     % Number of trials in split function
param.split = 'IG';     % Currently support 'information gain' only

trees = growTrees(data_Train,param); % this is the final function to be used for tree training

%% Test Random Forest

leaf_assign = testTrees_fast(data_Test,trees);

for T = 1:length(trees)
    p_rf(:,:,uint8(T)) = trees(1).prob(leaf_assign(:,uint8(T)),:);
end

% average the results from all trees
p_rf = squeeze(sum(p_rf,3))/length(trees); % Regression
[~,c] = max(p_rf'); % Regression to Classification
accuracy_rf = sum(c==data_Test(:,end)')/length(c); % Classification accuracy (for Caltech dataset)

disp(accuracy_rf);


















