function [image_names] = getImgnamesBySample(samples)
% given img folder, sampleImg folder and sample indicies
% return an cell array of image filenames, copy images to directory

% get parameters from samples

num_samples = size(samples,1);
image_names = cell(num_samples,1);

for i=1:num_samples
   weight = num2str(samples(i,1));
   content = num2str(samples(i,2));
   style  = num2str(samples(i,3));
   
   filename = strcat('weight',weight,'_content',content,'_style',style,'.png');
   image_names{i} = filename;
    
end


end