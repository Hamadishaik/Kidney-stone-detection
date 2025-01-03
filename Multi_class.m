data = load('Last.mat');
hmm = data.hmm;
%% 


rng(0)
shuffledIndices = randperm(height(hmm));

idx = floor(0.6 * height(hmm));

trainingIdx = 1:idx;
trainingDataTbl = hmm(shuffledIndices(trainingIdx),:);

validationIdx = idx+1 : idx + 1 + floor(0.1 * length(shuffledIndices) );
validationDataTbl = hmm(shuffledIndices(validationIdx),:);

testIdx = validationIdx(end)+1 : length(shuffledIndices);
testDataTbl = hmm(shuffledIndices(testIdx),:);

imdsTrain = imageDatastore(trainingDataTbl{:,'imageFilename'});
bldsTrain = boxLabelDatastore(trainingDataTbl(:,2:end));


imdsValidation = imageDatastore(validationDataTbl{:,'imageFilename'});
bldsValidation = boxLabelDatastore(validationDataTbl(:,2:end));

imdsTest = imageDatastore(testDataTbl{:,'imageFilename'});
bldsTest = boxLabelDatastore(testDataTbl(:,2:end));

trainingData = combine(imdsTrain,bldsTrain);
validationData = combine(imdsValidation,bldsValidation);
testData = combine(imdsTest,bldsTest);
%% 

data = read(trainingData);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

inputSize = [391 320 3];

preprocessedTrainingData = transform(trainingData, @(data)preprocessData(data,inputSize));
numAnchors = 5;
anchorBoxes = estimateAnchorBoxes(preprocessedTrainingData,numAnchors)

featureExtractionNetwork = resnet50;

featureLayer = 'activation_40_relu';

% %numClasses = 3;

% lgraph = fasterRCNNLayers(inputSize,numClasses,anchorBoxes,featureExtractionNetwork,featureLayer);


augmentedTrainingData = transform(trainingData,@augmentData);

augmentedData = cell(4,1);
for k = 1:4
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1},'rectangle',data{2});
    reset(augmentedTrainingData);
end
figure
montage(augmentedData,'BorderSize',30)

trainingData = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));
validationData = transform(validationData,@(data)preprocessData(data,inputSize));

data = read(trainingData);
%% 

I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)
%% 

%% 

options = trainingOptions('sgdm',...
    'MaxEpochs',20,...
    'MiniBatchSize',2,...
    'InitialLearnRate',1e-3,...
    'Plots','training-progress',...
    'ExecutionEnvironment','auto');
    %% 

detector = trainFasterRCNNObjectDetector(trainingData,lgraph,options, ...
        'NegativeOverlapRange',[0 0.3],...
        'PositiveOverlapRange',[0.6 1]);
%% 
I = imread(testDataTbl.imageFilename{65});
I = imresize(I,inputSize(1:2));
[bboxes,scores,labels] = detect(detector,I);

annotations=string(labels) +":" +string(scores);
I = insertObjectAnnotation(I,'Rectangle',bboxes,cellstr(annotations));


 % Display the image with the title
 figure
 imshow(I)
 %% 
  
 %% 
 testData = transform(testData,@(data)preprocessData(data,inputSize));
%% 
detectionResults = detect(detector,testData,...
    Threshold= 0.1,...
    MiniBatchSize=2);  
    %% 
classID = 3;
metrics = evaluateObjectDetection(detectionResults,testData);
precision = metrics.ClassMetrics.Precision{classID};
recall = metrics.ClassMetrics.Recall{classID};

c= metrics.ConfusionMatrix;
%% 


%% 
disp(c);
figure;
heatmap(c);
confusionchart(c,{'Normal','Stone','With_stone'});
xlabel('Actual-Labels');
ylabel('Predicted- Labels');
title('Confusion Matrix');
%% 

%% 
figure;
plot(recall, precision);
grid on
title(sprintf("Average Precision = %.2f", metrics.ClassMetrics.mAP(classID)))
xlabel('Recall');
ylabel('Precission');

 %% 
 image = I;

% Detect the object and get bounding box coordinates (assuming you have them)
bbox1 = bboxes; % Adjust according to your bounding box

border_width = 5; % Adjust the border width as needed
bbox_adjusted = [bbox1(1) + border_width, bbox1(2) + border_width, bbox1(3) - 2*border_width, bbox1(4) - 2*border_width];


% Extract the region of interest (ROI)
roi = imcrop(image, bbox_adjusted);

% Convert the ROI to grayscale if necessary
if size(roi, 3) == 3
    roi_gray = rgb2gray(roi);
else
    roi_gray = roi;
end

% Convert the grayscale ROI to a binary image using thresholding
threshold = graythresh(roi_gray);
binary_roi = imbinarize(roi_gray, threshold);

% Find the number of white pixels in each column
white_pixels_per_column = sum(binary_roi, 1);

% Find the column index with the maximum number of white pixels
[max_white_pixels_column_count, max_white_pixels_column_index] = max(white_pixels_per_column);

% Find the number of white pixels in each row
white_pixels_per_row = sum(binary_roi, 2);

% Find the row index with the maximum number of white pixels
[max_white_pixels_row_count, max_white_pixels_row_index] = max(white_pixels_per_row);

num_white_pixels = sum(binary_roi(:) == 1);
% Assuming you have physical dimensions (pixels per millimeter)
pixels_per_mm = 0.264; % Adjust according to your actual data

% Convert size of all white pixels from pixels to millimeters
area_white_pixels_mm2 = num_white_pixels*pixels_per_mm;

heigth = max_white_pixels_column_count*pixels_per_mm;
width= max_white_pixels_row_count*pixels_per_mm;

% Display the number of white pixels and the size of those pixels in millimeters
disp(['Number of white pixels in the binary ROI: ' num2str(num_white_pixels)]);
disp(['Area of all white pixels in the binary ROI (mm): ' num2str(area_white_pixels_mm2)]);
% Display results
disp(['Column with max white pixels: ', num2str(max_white_pixels_column_index)]);
disp(['Count of white pixels in max column: ', num2str(max_white_pixels_column_count)]);
disp(['Row with max white pixels: ', num2str(max_white_pixels_row_index)]);
disp(['Count of white pixels in max row: ', num2str(max_white_pixels_row_count)]);

disp(['The height of the stone in mm : ', num2str(heigth)]);
disp(['The width of the stone in mm: ', num2str(width)]);

 
%% 
% Display the original image and ROI
figure;
imshow(image);
hold on;
rectangle('Position', bbox1, 'EdgeColor', 'r', 'LineWidth', 2);
hold off;

% Display the ROI
figure;
imshow(roi);
title('Region of Interest');

figure;
imshow(roi_gray);
title('ROI- GRAYSCALE')
% Display the binary ROI
figure;
imshow(binary_roi);
title('Binary ROI');
figure;
subplot(1,2,1);
imshow(roi_gray);
subplot(1,2,2);
imshow(binary_roi);

%% 

function data = augmentData(data)
    % Randomly flip images and bounding boxes horizontally.
    tform = randomAffine2d('XReflection',true);
    sz = size(data{1});
    rout = affineOutputView(sz,tform);
    data{1} = imwarp(data{1},tform,'OutputView',rout);

    % Warp boxes.
    data{2} = bboxwarp(data{2},tform,rout);
    
    % Ensure bounding boxes are within image boundaries.
    data{2} = clipBBoxes(data{2}, sz);
end

function data = preprocessData(data,targetSize)
    % Resize image and bounding boxes to targetSize.
    sz = size(data{1},[1 2]);
    scale = targetSize(1:2)./sz;
    data{1} = imresize(data{1},targetSize(1:2));

    % Resize boxes.
    data{2} = bboxresize(data{2},scale);
    
    % Ensure bounding boxes are within image boundaries.
    data{2} = clipBBoxes(data{2}, targetSize);
end

function bboxes = clipBBoxes(bboxes, imageSize)
    % Clip bounding boxes to image boundaries.
    bboxes(:,1:2) = max(bboxes(:,1:2), [1 1]);
    bboxes(:,3) = min(bboxes(:,3), imageSize(2));
    bboxes(:,4) = min(bboxes(:,4), imageSize(1));
    
    % Remove boxes with invalid dimensions (negative width or height).
    validIdx = all(bboxes(:,3:4) > 0, 2);
    bboxes = bboxes(validIdx, :);
end


