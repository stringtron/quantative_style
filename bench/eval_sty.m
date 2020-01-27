
addpath benchmarks
clear all;close all;clc;

% setting evaluation parameters
PERM_GT = false;
isBoosting = false;
boosting_iter = 1000;

img_dir = '../BSDS500/data/images';
rstDir = '../BSDS500/ucm2/';

% stylized_img_dir = {'stang_test_Gatys','stang_test_Ours'}; 
% stylized_img_dir = {'stang_test_Gatys'}; 
stylized_img_dir = {'stang_test_AddCross50'}; 

for i = 1:length(stylized_img_dir)
    disp(['running on method: ',stylized_img_dir{i}]);

    if PERM_GT ==0
        outDir = fullfile(rstDir,strcat(stylized_img_dir{i},'_eval'));
        eval_sty_outDir = fullfile(rstDir,strcat(stylized_img_dir{i},'_eval_sum'));
    else
        outDir = fullfile(rstDir,strcat(stylized_img_dir{i},'_perm_eval'));
        eval_sty_outDir = fullfile(rstDir,strcat(stylized_img_dir{i},'_perm_eval_sum'));
    end
    mkdir(eval_sty_outDir)
    
    % get all eval txt file directories
    
    dirinfo = dir(outDir);
    dirinfo(~[dirinfo.isdir]) = [];  %remove non-directories
    dirinfo(ismember({dirinfo.name}, {'.', '..'})) = [];
    subdirinfo = cell(length(dirinfo), 1);
    numFile = 0;
    for k = 1 : length(dirinfo)
        thisdir = dirinfo(k).name;
        subdirinfo{k} = dir(fullfile(outDir,thisdir, '*.txt'));
        numFile = numFile + length(subdirinfo{k});
    end

   % bind file directory together 
   all_file_dir = cell(numFile,1);
   cnt = 1;
   for k = 1:length(dirinfo)
       thisdir = dirinfo(k).name;
       for n=1:length(subdirinfo{k})
        filename = subdirinfo{k}(n).name;
        all_file_dir{cnt} = fullfile(outDir,thisdir, filename);
        cnt = cnt+1;
       end
   end
    
    % evaluate each style
    
   iter =1;

   
   if isBoosting, iter = boosting_iter;end 
   sum_AUC = zeros(iter,1); 
   
   for k=1:iter
       fprintf('iter: %d\n',k)
        % randomly sample image to evaluate, sort of like boosting
        if isBoosting,
            iids = cell(numFile,1);
            
            for j = 1:numel(all_file_dir),      
                iids{j} = all_file_dir{randi(length(all_file_dir))};
            end
        else
            iids = all_file_dir;
        end
        
        % run style eval bdry
        tic;
        [bestF, bestP, bestR, bestT, F_max, P_max, R_max, Area_PR] = collect_eval_bdry_sty(eval_sty_outDir,iids);
        toc;
        
        sum_AUC(k) = Area_PR;
        
        
        % store auc value under this evaluation
        fname = fullfile(eval_sty_outDir,'eval_bdry_sum.txt');
        fid = fopen(fname,'a');
        if fid==-1,
            error('Could not open file %s for writing.',fname);
        end
        fprintf(fid,'%10g %10g %10g %10g %10g %10g %10g %10g\n',bestT,bestR,bestP,bestF,R_max,P_max,F_max,Area_PR);
        fclose(fid);
        
        %clear up rst if using boosting evaluation
        if isBoosting,
            system(sprintf('rm -f %s/eval_bdry.txt',eval_sty_outDir));
            system(sprintf('rm -f %s/eval_bdry_img.txt',eval_sty_outDir));
            system(sprintf('rm -f %s/eval_bdry_thr.txt',eval_sty_outDir));
        end
   end 
   
   % quantize boosting results
   if isBoosting,
       save([stylized_img_dir{i} '_1000.mat'],'sum_AUC')
       mean(sum_AUC)
       std(sum_AUC)
   else
       mean(sum_AUC)
       std(sum_AUC)
   end
   
end


