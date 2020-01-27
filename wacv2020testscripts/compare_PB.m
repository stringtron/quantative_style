
clear all
clc

store_dir = './rsts';
mkdir(store_dir)

isBoosting = false;
boosting_iter = 1;

sample_file = './wcs1.txt';
this_trial = 'round1';

img_dir = '../BSDS500/data/images';
cotent_Dir = '../BSDS500/ucm2/test';
rstDir = '../BSDS500/ucm2/SampleTests';



method1 = 'Gatys';
imgFolder1 = 'Gatysampled15';
SampleImgDir1 = fullfile(img_dir,imgFolder1);
rstDir1 = fullfile(rstDir,method1);
PbOutDir1 = fullfile(rstDir1,'PbRsts');

method2 = 'CrossLayer';
imgFolder2 = 'AddCrossampled15';
SampleImgDir2 = fullfile(img_dir,imgFolder2);
rstDir2 = fullfile(rstDir,method2);
PbOutDir2 = fullfile(rstDir2,'PbRsts');


samples = dlmread(sample_file);
image_names = getImgnamesBySample(samples);

all_file_dir1=cell(length(image_names),1);
for i =1:numel(image_names),
    PbOutFile = fullfile(PbOutDir1,[image_names{i}(1:end-4) '.mat']); 
    all_file_dir1{i} = PbOutFile;
end


all_file_dir2=cell(length(image_names),1);
for i =1:numel(image_names),
    PbOutFile = fullfile(PbOutDir2,[image_names{i}(1:end-4) '.mat']); 
    all_file_dir2{i} = PbOutFile;
end

content_files = cell(length(image_names),1);
for i = 1:numel(all_file_dir1),
    filename_splits = strsplit(all_file_dir1{i},'_');
    content_file_name = strrep(filename_splits{2},'content',''); 
    content_files{i} = fullfile(cotent_Dir, strcat(content_file_name,'.mat'));

end

% read mat files, plot and save

for i =1:numel(image_names),
    gatys_pb = load(all_file_dir1{i});
    crosslayer_pb = load(all_file_dir2{i});
    content_file = load(content_files{i});
    
    rgb = zeros(size(gatys_pb.ucm2,1),size(gatys_pb.ucm2,2),3 );
    rgb(:,:,1) = content_file.ucm2;
    rgb(:,:,2) = gatys_pb.ucm2;
    rgb(:,:,3) = crosslayer_pb.ucm2;
    
%     imshow(rgb)
    filename = strsplit(all_file_dir2{i},'/');
    imwrite(rgb,fullfile(store_dir,strrep(filename{7},'.mat','.png')))
    
    
end


