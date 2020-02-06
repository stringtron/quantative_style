


% move results to desinated file
imgRootDir = '../BSDS500/data/images';
inRootDir = '../BSDS500/ucm2';

imgDirArray = {'test_Gatys', 'test_Ours'};  

%content image pb mat
contentDir = '../BSDS500/ucm2/test';
% store dir
StoreContDir = './data/styles2/contours';
StoreImgDir = './data/styles2/images';
StoreCompareDir = './data/styles2/compareContours';

all_files = [];

for j = 1:numel(imgDirArray),
methodType = imgDirArray{j};
inDir = fullfile(inRootDir,methodType);
imgDir = fullfile(imgRootDir,methodType);
D = dir(fullfile(inDir,'*.mat'));
all_files = [all_files, {D.name}];
for i =1:numel(D),
    
    % pb file
    storefile = fullfile(StoreContDir,[D(i).name(1:end-4) '_' methodType '.bmp']);    
    matfile = load(fullfile(inDir,D(i).name));
    ucm2_style = matfile.ucm2;
    imwrite(ucm2_style,storefile);
    
    % image files
    storeImgfile = fullfile(StoreImgDir,[D(i).name(1:end-4) '_' methodType '.jpg']);
    img = imread(fullfile(imgDir,[D(i).name(1:end-4) '.jpg']));
    imwrite(img,storeImgfile)
    
    
end
end

%store content pb and images
for k = 1:numel(all_files),
    inDir = fullfile(inRootDir,'test');
    imgDir = fullfile(imgRootDir,'test');
    
    fileName = all_files{k};
    storefile = fullfile(StoreContDir,[fileName(1:end-4)  '.bmp']); 
    testFileName = strsplit(fileName,'_');
    testFileHead= strcat(testFileName{1},testFileName{2}(end-3:end));
    
    
    matfile = load(fullfile(inDir,testFileHead));
    ucm2 = matfile.ucm2;
    imwrite(ucm2,storefile);
    
    storeImgfile = fullfile(StoreImgDir,[fileName(1:end-4)  '.jpg']);
    img = imread(fullfile(imgDir,[testFileHead(1:end-4) '.jpg']));
    imwrite(img,storeImgfile) 
    
    
    % construct a rbg map of all pb maps
    matInDir = fullfile(inRootDir,'test_Ours');
    matGatysInDir = fullfile(inRootDir,'test_Gatys');

 
    matfile = load(fullfile(matInDir,fileName));
    ucm2_Ours = matfile.ucm2;
    matfile = load(fullfile(matGatysInDir,fileName));
    ucm2_Gatys = matfile.ucm2;

    Size = size(ucm2_Ours);
    img_rgb = zeros(Size(1),Size(2),3);
    img_rgb(:,:,1) = ucm2;
    img_rgb(:,:,2) = ucm2_Gatys;
    img_rgb(:,:,3) = ucm2_Ours;
    
    storeCompFile = fullfile(StoreCompareDir,[fileName(1:end-4)  '.bmp']);
    imwrite(img_rgb,storeCompFile);
    %imshow(img_rgb)

    
    
    
    
    
end
