function boundaryBench_sty(imgDir, gtDir, pbDir, outDir, isBoosting,nthresh, maxDist, thinpb)
% boundaryBench(imgDir, gtDir, pbDir, outDir, nthresh, maxDist, thinpb)
%
% Run boundary benchmark (precision/recall curve) on dataset.
% adpated for evaluate stylized images 
%
% INPUT
%   imgDir: folder containing original images
%   gtDir:  folder containing ground truth data.
%   pbDir:  folder containing boundary detection results for all the images in imgDir. 
%           Format can be one of the following:
%             - a soft or hard boundary map in PNG format.
%             - a collection of segmentations in a cell 'segs' stored in a mat file
%             - an ultrametric contour map in 'doubleSize' format, 'ucm2' stored in a mat file with values in [0 1].
%   outDir: folder where evaluation results will be stored
%	nthresh	: Number of points in precision/recall curve.
%   MaxDist : For computing Precision / Recall.
%   thinpb  : option to apply morphological thinning on segmentation
%             boundaries before benchmarking.
%
% based on boundaryBench by David Martin and Charless Fowlkes:
% http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/code/Benchmark/boundaryBench.m
%
% Pablo Arbelaez <arbelaez@eecs.berkeley.edu>

if nargin<8, thinpb = true; end
if nargin<7, maxDist = 0.0075; end
if nargin<6, nthresh = 99; end
% additional flag to random sample images
if nargin<5, isBoosting = false; end


iids_raw = dir(fullfile(imgDir,'*.jpg'));

% randomly sample image to evaluate, sort of like boosting
if isBoosting,
    iids(1:length(iids_raw)) = struct('name',[],'date',[],'bytes',[],'isdir',[],'datenum',[]);
    iids = iids';
    for j = 1:numel(iids_raw),      
        iids(j) = iids_raw(randi(length(iids_raw)));
    end
else
    iids = iids_raw;
end

for i = 1:numel(iids),
    fprintf('processing image: %s (%d/%d) \n',iids(i).name,i,numel(iids))
    
    evFile = fullfile(outDir, strcat(iids(i).name(1:end-4),'_ev1.txt'));
    
    if exist(evFile,'file'), 
        disp('find duplicate image')
        S = dir(fullfile(outDir, strcat(iids(i).name(1:end-4),'_*')));
        num = length(S)+1;
        evFile_new = fullfile(outDir, strcat(iids(i).name(1:end-4),'_',num2str(num),'_ev1.txt'));
        copyfile(evFile,evFile_new)
        continue;
    end
    
    inFile = fullfile(pbDir, strcat(iids(i).name(1:end-4),'.mat'));
    if ~exist(inFile,'file'),
        inFile = fullfile(pbDir, strcat(iids(i).name(1:end-4),'.png'));
    end
    gtFile = fullfile(gtDir, strcat(iids(i).name(1:end-4),'.mat'));
    evaluation_bdry_image(inFile,gtFile, evFile, nthresh, maxDist, thinpb);
    
end

%% collect results
% collect_eval_bdry(outDir);
% 
% % store auc value under this evaluation
% 
% eval_rst = dlmread(fullfile(outDir,'eval_bdry.txt'));
% 
% fname = fullfile(outDir,'eval_bdry_sum.txt');
% fid = fopen(fname,'a');
% if fid==-1,
%     error('Could not open file %s for writing.',fname);
% end
% fprintf(fid,'%10g %10g %10g %10g %10g %10g %10g %10g\n',eval_rst(1),eval_rst(2),eval_rst(3),eval_rst(4),eval_rst(5),eval_rst(6),eval_rst(7),eval_rst(8));
% fclose(fid);

%% clean up
% system(sprintf('rm -f %s/*_ev1.txt',outDir));
% system(sprintf('rm -f %s/eval_bdry.txt',outDir));
% system(sprintf('rm -f %s/eval_bdry_img.txt',outDir));
% system(sprintf('rm -f %s/eval_bdry_thr.txt',outDir));






