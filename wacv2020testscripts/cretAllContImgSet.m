clc; clear

% create calibration experiment one. style transfer split original content

method = 'allContent';
imgFolder = 'AllContSampled';
originalImgFolder = 'test';
img_dir = '../BSDS500/data/images';

sample_file = './wcs1.txt';
samples = dlmread(sample_file);
image_names = getImgnamesBySample(samples); % return an cell array of image filenames, copy images to directory

original_img_dir = fullfile(img_dir,originalImgFolder);
target_img_dir = fullfile(img_dir,imgFolder);
mkdir(target_img_dir);

for i = 1:length(image_names)
   tarImgFilename =  fullfile(target_img_dir,image_names{i});
   originalFilename = fullfile(original_img_dir,strcat(num2str(samples(i,2)),'.jpg'));
   im = imread(originalFilename);   
   imwrite(im,tarImgFilename)
end