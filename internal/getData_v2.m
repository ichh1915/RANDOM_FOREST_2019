function [ data_train, data_query ] = getData( MODE )
% Generate training and testing data

% Data Options:
%   1. Toy_Gaussian
%   2. Toy_Spiral
%   3. Toy_Circle
%   4. Caltech 101

showImg = 0; % Show training & testing images and their image feature vector (histogram representation)

PHOW_Sizes = [4 8 10]; % Multi-resolution, these values determine the scale of each layer.
PHOW_Step = 8; % The lower the denser. Select from {2,4,8,16}

switch MODE
    case 'Toy_Gaussian' % Gaussian distributed 2D points
        %rand('state', 0);
        %randn('state', 0);
        N= 150;
        D= 2;
        
        cov1 = randi(4);
        cov2 = randi(4);
        cov3 = randi(4);
        
        X1 = mgd(N, D, [randi(4)-1 randi(4)-1], [cov1 0;0 cov1]);
        X2 = mgd(N, D, [randi(4)-1 randi(4)-1], [cov2 0;0 cov2]);
        X3 = mgd(N, D, [randi(4)-1 randi(4)-1], [cov3 0;0 cov3]);
        
        X= real([X1; X2; X3]);
        X= bsxfun(@rdivide, bsxfun(@minus, X, mean(X)), var(X));
        Y= [ones(N, 1); ones(N, 1)*2; ones(N, 1)*3];
        
        data_train = [X Y];
        
    case 'Toy_Spiral' % Spiral (from Karpathy's matlab toolbox)
        
        N= 50;
        t = linspace(0.5, 2*pi, N);
        x = t.*cos(t);
        y = t.*sin(t);
        
        t = linspace(0.5, 2*pi, N);
        x2 = t.*cos(t+2);
        y2 = t.*sin(t+2);
        
        t = linspace(0.5, 2*pi, N);
        x3 = t.*cos(t+4);
        y3 = t.*sin(t+4);
        
        X= [[x' y']; [x2' y2']; [x3' y3']];
        X= bsxfun(@rdivide, bsxfun(@minus, X, mean(X)), var(X));
        Y= [ones(N, 1); ones(N, 1)*2; ones(N, 1)*3];
        
        data_train = [X Y];
        
    case 'Toy_Circle' % Circle
        
        N= 50;
        t = linspace(0, 2*pi, N);
        r = 0.4
        x = r*cos(t);
        y = r*sin(t);
        
        r = 0.8
        t = linspace(0, 2*pi, N);
        x2 = r*cos(t);
        y2 = r*sin(t);
        
        r = 1.2;
        t = linspace(0, 2*pi, N);
        x3 = r*cos(t);
        y3 = r*sin(t);
        
        X= [[x' y']; [x2' y2']; [x3' y3']];
        Y= [ones(N, 1); ones(N, 1)*2; ones(N, 1)*3];
        
        data_train = [X Y];
%========================================================================================================%        
    case 'Caltech' % Caltech dataset
        close all;
        %imgSel = [15 15]; % randomly select 15 images each class without replacement. (For both training & testing)
        folderName = './Caltech_101/101_ObjectCategories';
        classList = dir(folderName);
        classList = {classList(3:end).name} % 10 classes
        
        disp('Loading training images...')
        % Load Images -> Description (Dense SIFT)
        cnt = 1;
        if showImg
            figure('Units','normalized','Position',[.05 .1 .4 .9]);
            suptitle('Training image samples');
        end
        for c = 1:length(classList)
            subFolderName = fullfile(folderName,classList{c});
            imgList = dir(fullfile(subFolderName,'*.jpg')); % select all the images for this class
            imgIdx{c} = randperm(length(imgList)); % random permutation n: e.g. n-2 3 n-5 1 .., where n>30 is the number of images in the class
            imgIdx_tr = imgIdx{c}(1:round(length(imgList)*0.7)); % idx for training
            
            %imgIdx_te =%imgIdx{c}(round(length(imgList)*0.7)+1:length(imgList)); 
            %imgIdx_tr = imgIdx{c}(1:imgSel(1)); % idx 1-15 for training
            %imgIdx_te = imgIdx{c}(imgSel(1)+1:sum(imgSel)); % idx 16-30 for testing
            
            for i = 1:length(imgIdx_tr) % from all training images
                I = imread(fullfile(subFolderName,imgList(imgIdx_tr(i)).name));
                
                % Visualise
%                 if i < 6 & showImg
%                     subaxis(length(classList),5,cnt,'SpacingVert',0,'MR',0);
%                     imshow(I);
%                     cnt = cnt+1;
%                     drawnow;
%                 end
                
                if size(I,3) == 3 % if the 3rd dim of I is 3 - check if image is in RGB 
                    I = rgb2gray(I); % PHOW work on gray scale image % convert RGB image to grayscale image
                end
                
                % For details of image description, see http://www.vlfeat.org/matlab/vl_phow.html
                [~, desc_tr{c,i}] = vl_phow(single(I),'Sizes',PHOW_Sizes,'Step',PHOW_Step); %  extracts PHOW features (multi-scaled Dense SIFT)
                % vl_phow() extracts PHOW features from the image
            
            end
        end
        
        disp('Building visual codebook...')
        % Build visual vocabulary (codebook) for 'Bag-of-Words method'
        desc_sel = single(vl_colsubset(cat(2,desc_tr{:}),1e3)); % Randomly select 100k (1e5) SIFT descriptors for clustering
        % vl_colsubset returns a random subset Y of N columns of X
        
        
        %profile -memory on
        % K-means clustering
        disp('K-means clustering...')
        num_centers = 32; % for instance,
        %[idx,C] = kmeans(desc_sel.',num_centers,'MaxIter',1e10,'Start','cluster');
        [C,idx]=vl_ikmeans(uint8(desc_sel),num_centers);
        % [C,idx]=vl_hikmeans(uint8(desc_sel),num_centers);
        Kmean_index = idx;
        Kmean_centers = C;
        %profile viewer  
        
        disp('Encoding Images...')
        % Vector Quantisation for the training data, stored by class
      
        % hist_tr = zeros(length(classList),239,num_centers); % 10 * 239 * (32 * 1)
        hist_tr=[];
        y_tr=[];
        for c = 1:length(classList) %1:10
            Length_imgList_c = length(imgIdx{c});
            imgIdx_tr = imgIdx{c}(1:round(Length_imgList_c*0.7)); % idx for training
            for i = 1:length(imgIdx_tr) %1:length(imgIdx{c})  
                desc_ci = desc_tr{c,i};
                [~,NoP_ci] = size(desc_ci);
                % for p = 1:NoP_ci
                    %desc_ci_p = desc_ci(:,p); % each of p visual words is classified by NN with the codebook (k-mean cluster centres)
                Idx_ci_p = knnsearch(Kmean_centers.',desc_ci.');
                %Idx_ci_p = knnsearch(Kmean_centers,desc_ci_p);
                hist_tr=[hist_tr,vl_ikmeanshist(32,Idx_ci_p)]; % bin the visual word to the nearest codeword  
                y_tr=[y_tr,c];
            end 
        end
        data_train = [hist_tr.' y_tr.'];
        
        % Clear unused varibles to save memory
        clearvars desc_tr desc_sel
end

switch MODE
    case 'Caltech'
        if showImg
        figure('Units','normalized','Position',[.05 .1 .4 .9]);
        suptitle('Test image samples');
        end
        disp('Processing testing images...');
        cnt = 1;
        % Load Images -> Description (Dense SIFT)
        for c = 1:length(classList)
            subFolderName = fullfile(folderName,classList{c});
            imgList = dir(fullfile(subFolderName,'*.jpg'));
            imgIdx_te = imgIdx{c}(round(length(imgList)*0.7)+1:length(imgList)); 
            % imgIdx_te = imgIdx{c}(imgSel(1)+1:sum(imgSel));
            
            for i = 1:length(imgIdx_te)
                I = imread(fullfile(subFolderName,imgList(imgIdx_te(i)).name));
                
                % Visualise
%                 if i < 6 & showImg
%                     subaxis(length(classList),5,cnt,'SpacingVert',0,'MR',0);
%                     imshow(I);
%                     cnt = cnt+1;
%                     drawnow;
%                 end
                
                if size(I,3) == 3
                    I = rgb2gray(I);
                end
                [~, desc_te{c,i}] = vl_phow(single(I),'Sizes',PHOW_Sizes,'Step',PHOW_Step);
            
            end
        end
        %suptitle('Testing image samples');
%                 if showImg
%             figure('Units','normalized','Position',[.5 .1 .4 .9]);
%         suptitle('Testing image representations: 256-D histograms');
%         end

        % Quantisation       
        hist_te=[];
        y_te=[];
        for c = 1:length(classList) %1:10
            Length_imgList_c = length(imgIdx{c});
            imgIdx_te = imgIdx{c}(round(Length_imgList_c*0.7)+1:Length_imgList_c);  % idx for training
            for i = 1:length(imgIdx_te) %1:length(imgIdx{c})  
                desc_ci = desc_te{c,i};
                [~,NoP_ci] = size(desc_ci);
                % for p = 1:NoP_ci
                    %desc_ci_p = desc_ci(:,p); % each of p visual words is classified by NN with the codebook (k-mean cluster centres)
                Idx_ci_p = knnsearch(Kmean_centers.',desc_ci.');
                %Idx_ci_p = knnsearch(Kmean_centers,desc_ci_p);
                hist_te=[hist_te,vl_ikmeanshist(32,Idx_ci_p)]; % bin the visual word to the nearest codeword  
                y_te=[y_te,c];            
            end 
        end       
        data_query = [hist_te.' y_te.'];
   
        
    
    otherwise % Dense point for 2D toy data
        xrange = [-1.5 1.5];
        yrange = [-1.5 1.5];
        inc = 0.02;
        [x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
        data_query = [x(:) y(:) zeros(length(x)^2,1)];
end
end

