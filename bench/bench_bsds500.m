addpath benchmarks

clear all;close all;clc;

gtDir = '../BSDS500/data/groundTruth/test_selected';

imgDir = '../BSDS500/data/images/test_Ours';
inDir = '../BSDS500/ucm2/test_Ours';
outDir = '../BSDS500/ucm2/test_Ours_eval';

% imgDir = '../BSDS500/data/images/test_Gatys';
% inDir = '../BSDS500/ucm2/test_Gatys';
% outDir = '../BSDS500/ucm2/test_Gatys_eval';


mkdir(outDir);

% running all the benchmarks can take several hours.
tic;
% allBench(imgDir, gtDir, inDir, outDir)
boundaryBench(imgDir, gtDir, inDir, outDir)
toc;

plot_eval(outDir);



