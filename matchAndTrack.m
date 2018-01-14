clc, clear, close all;

%% Part 0: Prepare
% -Load reference images
refImage = imread('img1_mod.png');

% -Get features
refImageGray = rgb2gray(refImage);
refPts = detectSURFFeatures(refImageGray);
refFeatures = extractFeatures(refImageGray,refPts);
figure, imshow(refImage), hold on;
plot(refPts.selectStrongest(50));


% Create webcam object
cam = webcam

%% Part 1: Read frame
NUM_IMG = 2;
C = cell(NUM_IMG,1);
while(1)
    camFrame = snapshot(cam);


    %camFrame = imread('img2.png');
    camFrameGray = rgb2gray(camFrame);
    camPts = detectSURFFeatures(camFrameGray);
    camFeatures = extractFeatures(camFrameGray,camPts);
%     figure, imshow(camFrame), hold on;
%     plot(camPts.selectStrongest(50));


    %% Part 2: Recognize
    idxPairs = matchFeatures(camFeatures,refFeatures);

    % Store SURF points that were matched
    matchedCamPts = camPts(idxPairs(:,1));
    matchedRefPts = refPts(idxPairs(:,2));

    %% Part 3: Display
%     figure,
%     showMatchedFeatures(camFrame, refImage, ...
%                         matchedCamPts, matchedRefPts,'Montage');

    [tform, inlierRefPts, inlierCamPts] ...
        = estimateGeometricTransform(...
            matchedRefPts, matchedCamPts, 'Similarity');

%     % Show the inliers of the estimated geometric tranformation
%     figure,
%     showMatchedFeatures(camFrame, refImage, ...
%                         inlierCamPts, inlierRefPts,'Montage');

    %% Rescale replacement video frame
    outputView = imref2d(size(camFrameGray));
    Ir = imwarp(refImageGray,tform,'OutputView',outputView);
%     figure; imshow(Ir); 
%     title('tranformed image');

    %% Transform bounding bo from orig image to distorted image

    % Create bounding bo around orig image
    boxPolygon = [1, 1;...                           % top-left
            size(refImageGray, 2), 1;...                 % top-right
            size(refImageGray, 2), size(refImageGray, 1);... % bottom-right
            1, size(refImageGray, 1);...                 % bottom-left
            1, 1];                   % top-left again to close the polygon

    % Tranform bounding bo
    newBoxPolygon = transformPointsForward(tform, boxPolygon);

    figure(2);
    imshow(camFrame);
    hold on;
    line(newBoxPolygon(:, 1), newBoxPolygon(:, 2), 'Color', 'y');
    title('Detected Box');
end
