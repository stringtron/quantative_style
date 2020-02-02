addpath benchmarks
clear all;close all;clc;

% setting evaluation parameters
PERM_GT = false;
isBoosting = false;
bossting_iter = 2;

img_dir = '../BSDS500/data/images';
gtDir = '../BSDS500/data/groundTruth';
rstDir = '../BSDS500/ucm2/';
% stylized_img_dir = {'stang_test_Gatys','stang_test_Ours'}; 
stylized_img_dir = {'stang_test_AddCross50'}; 

% assert(length(dir(fullfile(img_dir,stylized_img_dir{1})))==length(dir(fullfile(img_dir,stylized_img_dir{2}))))
styles = {
          'style84'
          'style189'
          'style48'
          'style86' 
          'style89'   
          'style194'
          'style6' 
          'style85' 
          'style88' 
          'style95' 
          };


if PERM_GT == 0
    gtDir = fullfile(gtDir,'test_selected');
else
    gtDir = fullfile(gtDir,'test_selected_perm');
end
   
%evaluate each style transfer

%using parpool
for i = 1:length(stylized_img_dir)
    disp(['running on method: ',stylized_img_dir{i}]);
    imgDir = fullfile(img_dir,stylized_img_dir{i});
    inDir = fullfile(rstDir,stylized_img_dir{i});
    if PERM_GT ==0
        outDir = fullfile(rstDir,strcat(stylized_img_dir{i},'_eval'));
    else
        outDir = fullfile(rstDir,strcat(stylized_img_dir{i},'_perm_eval'));
    end
    mkdir(outDir);
    % evaluate each style
    parfor j = 1:length(styles)
       
       style = styles{j};
       disp(['processing on style: ',style]);
       imgDir_sty = fullfile(imgDir,style);
       inDir_sty = fullfile(inDir,style);
       outDir_sty = fullfile(outDir,style);
       mkdir(outDir_sty);
       % running all the benchmarks can take several hours.
       iter =1;
       if isBoosting, iter = bossting_iter;end
       for k=1:iter
           fprintf('iter: %d\n',k)
           tic;
           boundaryBench_sty(imgDir_sty, gtDir, inDir_sty, outDir_sty,isBoosting)
           toc;
       end
%        plot_eval(outDir_sty);
       
    end
    
end

% imgDir = 'test_Ours';
% inDir = 'test_Ours';
% outDir = 'test_Ours_eval';
% 
% % imgDir = '../BSDS500/data/images/test_Gatys';
% % inDir = '../BSDS500/ucm2/test_Gatys';
% % outDir = '../BSDS500/ucm2/test_Gatys_eval';
% mkdir(outDir);

% % running all the benchmarks can take several hours.
% tic;
% % allBench(imgDir, gtDir, inDir, outDir)
% boundaryBench_sty(imgDir, gtDir, inDir, outDir,isBoosting)
% toc;
% 
% plot_eval(outDir);


