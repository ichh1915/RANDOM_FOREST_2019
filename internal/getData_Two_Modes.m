function [ data_Train, data_Test ] = getData_Two_Modes( MODE_CodeBook )
% Initialise & load data
PHOW_Sizes = [4 8 10]; % Multi-resolution, these values determine the scale of each layer.
PHOW_Step = 8; % The lower the denser. Select from {2,4,8,16}
close all;
folderName = './Caltech_101/101_ObjectCategories';
classList = dir(folderName);
classList = {classList(3:end).name}; % 10 classes

% Load Images -> Description (Dense SIFT)
disp('Loading training images...')
disp('Applying SIFT...')


switch MODE_CodeBook
    case 'RF_Codebook'
        desc_Train_Img_Index = zeros(length(classList),1);
        for c = 1:length(classList)
            subFolderName = fullfile(folderName,classList{c});
            imgList = dir(fullfile(subFolderName,'*.jpg')); % select all the images for this class
            imgIdx{c} = randperm(length(imgList)); % random permutation 
            imgIdx_Train = imgIdx{c}(1:round(length(imgList)*0.7)); % idx for training data          
            imgIdx_Test = imgIdx{c}(round(length(imgList)*0.7)+1:length(imgList)); % idx for testing data         

            for i = 1:length(imgIdx_Train) % from all training images of class c
                I = imread(fullfile(subFolderName,imgList(imgIdx_Train(i)).name));
                % plot training sample c3 img1
                if c==3 && i==1
                    figure
                    imshow(I);
                end

                if size(I,3) == 3  
                    I = rgb2gray(I);
                end
                [~, desc_Train{c,i}] = vl_phow(single(I),'Sizes',PHOW_Sizes,'Step',PHOW_Step);       
                desc_length = size(desc_Train{c,i},2);
                desc_Train{c,i} = [desc_Train{c,i}; c*ones(1,desc_length)];
            end

            for i = 1:length(imgIdx_Test) % from all testing images of class c
                I = imread(fullfile(subFolderName,imgList(imgIdx_Test(i)).name));
                % plot testing sample c3 img1
                if c==4 && i==1
                    figure
                    imshow(I);
                end

                if size(I,3) == 3  
                    I = rgb2gray(I); 
                end
                [~, desc_Test{c,i}] = vl_phow(single(I),'Sizes',PHOW_Sizes,'Step',PHOW_Step);          
                desc_length = length(desc_Test{c,i});
                desc_Test{c,i} = [desc_Test{c,i}; c*ones(1,desc_length)];
            end
        end

        desc_Train_CodeBOOK = [];
        for c = 1:length(classList) 
            desc_Train_CodeBOOK = [desc_Train_CodeBOOK desc_Train{c,:}];
        end
        desc_Train_CodeBOOK = desc_Train_CodeBOOK.'; 

        % Training RF codebook
        param_CodeBOOK.num = 5;          % Number of trees % intially 50
        param_CodeBOOK.depth = 5;        % Depth of each tree
        param_CodeBOOK.splitNum = 5;     % Number of trials in split function
        param_CodeBOOK.split = 'IG';     % Currently support 'information gain' only

        disp('training RF Codebook...')
        trees_CodeBOOK = growTrees(desc_Train_CodeBOOK,param_CodeBOOK); % this is the final function to be used for tree training

        [numLeavesTotal,~] = size(trees_CodeBOOK(1).prob);
        hist_Train=[];
        hist_Test=[];
        hist_Train_re=[];
        hist_Test_re=[]; 

        for c = 1:length(classList)
            subFolderName = fullfile(folderName,classList{c});
            imgList = dir(fullfile(subFolderName,'*.jpg')); % select all the images for this class
            imgIdx{c} = randperm(length(imgList)); % random permutation 
            imgIdx_Train = imgIdx{c}(1:round(length(imgList)*0.7)); % idx for training data          
            imgIdx_Test = imgIdx{c}(round(length(imgList)*0.7)+1:length(imgList)); % idx for testing data         

            for i = 1:length(imgIdx_Train) % from all training images of class c
                leaf_assign_Train = testTrees_fast(desc_Train{c,i},trees_CodeBOOK);
                hist_Train(c,i,:)=histc(reshape(leaf_assign_Train,1,numel(leaf_assign_Train)),1:numLeavesTotal)./numel(leaf_assign_Train);
                % visualise hist for c3 img1
                if c==3 && i==1
                    visual_hist = hist_Train(c,i,:);
                    nbin = length(visual_hist);
                    figure;
                    histogram(visual_hist, nbin);
                end

                hist_Train(c,i,end)=c;
                hist_Train_re = [hist_Train_re hist_Train(c,i,:)];
            end

            for i = 1:length(imgIdx_Test) % from all training images of class c
                leaf_assign_Test = testTrees_fast(desc_Train{c,i},trees_CodeBOOK);
                hist_Test(c,i,:)=histc(reshape(leaf_assign_Test,1,numel(leaf_assign_Test)),1:numLeavesTotal)./numel(leaf_assign_Test);
                % visualise hist for c3 img1
                if c==4 && i==1
                    visual_hist = hist_Test(c,i,:);
                    nbin = length(visual_hist);
                    figure;
                    histogram(visual_hist, nbin);
                end

                hist_Test(c,i,end)=c;
                hist_Test_re = [hist_Test_re hist_Test(c,i,:)];
            end
        end

        % reshape to 2D matrix form
        [~,N_data_Train,~] = size(hist_Train_re);
        [~,N_data_Test,~] = size(hist_Test_re);
        data_Train = reshape(hist_Train_re,[N_data_Train numLeavesTotal]);
        data_Test = reshape(hist_Test_re,[N_data_Test numLeavesTotal]);
        
        
        
        
        
    case 'KMEAN_Codebook'
        close all;
        folderName = './Caltech_101/101_ObjectCategories';
        classList = dir(folderName);
        classList = {classList(3:end).name}; % 10 classes
        
        % Load Images -> Description (Dense SIFT)
        disp('Loading training images...')
        for c = 1:length(classList)
            subFolderName = fullfile(folderName,classList{c});
            imgList = dir(fullfile(subFolderName,'*.jpg')); % select all the images for this class
            imgIdx{c} = randperm(length(imgList)); % random permutation n: e.g. n-2 3 n-5 1 .., where n>30 is the number of images in the class
            imgIdx_tr = imgIdx{c}(1:round(length(imgList)*0.7)); % idx for training           
            for i = 1:length(imgIdx_tr) % from all training images
                I = imread(fullfile(subFolderName,imgList(imgIdx_tr(i)).name));
                if size(I,3) == 3 % if the 3rd dim of I is 3 - check if image is in RGB 
                    I = rgb2gray(I); % PHOW work on gray scale image % convert RGB image to grayscale image
                end
                [~, desc_tr{c,i}] = vl_phow(single(I),'Sizes',PHOW_Sizes,'Step',PHOW_Step); %  extracts PHOW features (multi-scaled Dense SIFT)
            end
        end
        disp('Building visual codebook...')
        desc_sel = single(vl_colsubset(cat(2,desc_tr{:}),5e4)); % Randomly select 100k (1e5) SIFT descriptors for clustering

        disp('K-means clustering...')
        num_centers = 256; 
        %[idx,C] = kmeans(desc_sel.',num_centers,'MaxIter',1e10,'Start','cluster');
        tic
        [C,~]=vl_ikmeans(uint8(desc_sel),num_centers);
        toc
        Kmean_centers = C;
        
        disp('Encoding Images...')
        hist_tr=[];
        y_tr=[];
        for c = 1:length(classList) %1:10
            Length_imgList_c = length(imgIdx{c});
            imgIdx_tr = imgIdx{c}(1:round(Length_imgList_c*0.7)); % idx for training
            for i = 1:length(imgIdx_tr) %1:length(imgIdx{c})  
                desc_ci = desc_tr{c,i};
                [~,NoP_ci] = size(desc_ci);
                Idx_ci_p = knnsearch(double(Kmean_centers.'),double(desc_ci.'));
                hist_tr=[hist_tr,vl_ikmeanshist(num_centers,Idx_ci_p)]; % bin the visual word to the nearest codeword  
                y_tr=[y_tr,c];
            end 
        end
        data_Train = [hist_tr.' y_tr.'];
        
        % Build { data_Test ]
        for c = 1:length(classList)
            subFolderName = fullfile(folderName,classList{c});
            imgList = dir(fullfile(subFolderName,'*.jpg'));
            imgIdx_te = imgIdx{c}(round(length(imgList)*0.7)+1:length(imgList)); 
            % imgIdx_te = imgIdx{c}(imgSel(1)+1:sum(imgSel));
            
            for i = 1:length(imgIdx_te)
                I = imread(fullfile(subFolderName,imgList(imgIdx_te(i)).name));
                if size(I,3) == 3
                    I = rgb2gray(I);
                end
                [~, desc_te{c,i}] = vl_phow(single(I),'Sizes',PHOW_Sizes,'Step',PHOW_Step);
            
            end
        end
        % Quantisation       
        hist_te=[];
        y_te=[];
        for c = 1:length(classList) %1:10
            Length_imgList_c = length(imgIdx{c});
            imgIdx_te = imgIdx{c}(round(Length_imgList_c*0.7)+1:Length_imgList_c);  % idx for training
            for i = 1:length(imgIdx_te) %1:length(imgIdx{c})  
                desc_ci = desc_te{c,i};
                [~,NoP_ci] = size(desc_ci);
                Idx_ci_p = knnsearch(double(Kmean_centers.'),double(desc_ci.'));
                hist_te=[hist_te,vl_ikmeanshist(num_centers,Idx_ci_p)]; % bin the visual word to the nearest codeword  
                y_te=[y_te,c];            
            end 
        end       
        data_Test = [hist_te.' y_te.'];
        % Clear unused varibles to save memory
        clearvars desc_tr desc_sel
end







