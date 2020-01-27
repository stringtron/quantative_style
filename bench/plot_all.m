addpath benchmarks
clear all;close all;clc;


Gatys_out = '../BSDS500/ucm2/test_Gatys_eval';
Ours_out = '../BSDS500/ucm2/test_Ours_eval';


rst_dirs = {Gatys_out
            Ours_out
           '../BSDS500/ucm2/stang_test_Gatys_perm_eval_sum'
           '../BSDS500/ucm2/stang_test_Ours_perm_eval_sum'
           };
       
% plot gatys results
open('isoF.fig');
hold on


% plot Gatys rsts
prvals = dlmread(fullfile(Gatys_out,'eval_bdry_thr.txt')); % thresh,r,p,f
f=find(prvals(:,2)>=0.01);
prvals = prvals(f,:);

evalRes = dlmread(fullfile(Gatys_out,'eval_bdry.txt'));

if size(prvals,1)>1,
    gatys = plot(prvals(1:end,2),prvals(1:end,3),'r','LineWidth',3,'DisplayName',sprintf('Gatys(AUC=%1.3f)',evalRes(8)));
else
   print 'gatys results not valid'; 
end


% plot ours rsts
prvals = dlmread(fullfile(Ours_out,'eval_bdry_thr.txt')); % thresh,r,p,f
f=find(prvals(:,2)>=0.01);
prvals = prvals(f,:);

evalRes = dlmread(fullfile(Ours_out,'eval_bdry.txt'));
if size(prvals,1)>1,
    ours = plot(prvals(1:end,2),prvals(1:end,3),'b','LineWidth',3,'DisplayName',sprintf('Ours(AUC=%1.3f)',evalRes(8)));
else
   print 'ours results not valid'; 
end


% gatys perm

prvals = dlmread(fullfile(rst_dirs{3},'eval_bdry_thr.txt')); % thresh,r,p,f
f=find(prvals(:,2)>=0.01);
prvals = prvals(f,:);

evalRes = dlmread(fullfile(rst_dirs{3},'eval_bdry.txt'));

if size(prvals,1)>1,
    gatys_perm = plot(prvals(1:end,2),prvals(1:end,3),'-.r','LineWidth',3,'DisplayName',sprintf('Gatys randGT(AUC=%1.3f)',evalRes(8)));
else
   print 'gatys results not valid'; 
end



% ours perm
prvals = dlmread(fullfile(rst_dirs{4},'eval_bdry_thr.txt')); % thresh,r,p,f
f=find(prvals(:,2)>=0.01);
prvals = prvals(f,:);

evalRes = dlmread(fullfile(rst_dirs{4},'eval_bdry.txt'));

if size(prvals,1)>1,
    ours_perm = plot(prvals(1:end,2),prvals(1:end,3),'-.b','LineWidth',3,'DisplayName',sprintf('Ours randGT(AUC=%1.3f)',evalRes(8)));
else
   print 'gatys results not valid'; 
end





legend([gatys, ours, gatys_perm,ours_perm],'Location','northwest')