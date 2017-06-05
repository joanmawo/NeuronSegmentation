% Script that dowloads, unzips, and expands annotations for training DRIU's CNN for neuron segmentation.
% 

dataSet = '0003';
cropSize = 400; % Defined final size of all images

ORIGIN = strcat(dataSet);
IMAGES = strcat(ORIGIN,'/Images');
CROPS = strcat('crops/images');
ANNOTATIONS = strcat('crops/annotations');

%% DATA DOWNLOAD %%%
% Downloads the complete folder with training images, 
 outname = websave(strcat(dataSet,'.zip'),'https://www.dropbox.com/s/mmt7vzcp7e3jacm/0003.zip?dl=1');
 %%% removal https://1fichier.com/remove/ay5dxaya66/oAXAL
% unzip(strcat(dataSet,'.zip'));
unzip(outname);

mkdir crops
mkdir('crops/images')
mkdir('crops/annotations')
mkdir CROPS
mkdir ANNOTATIONS 

%% DATA AND ANNOTATION EXPANSION
% Get annotations
ann_name = strcat(ORIGIN,'/Annotations/neurofinder.00.03.tiff');
ann = imread(ann_name); 
[row col] = size(ann);
ann = im2double(ann);

% Create new image to save the 3-channel intensity-sum image (100%, 20%, 5%)
new_image = zeros(row, col, 3);

% Sum intensity along time axis
ims_names = dir(strcat(IMAGES,'/*.tiff'));

im = imread(strcat(IMAGES,'/',ims_names(1).name));

video = zeros(row, col, length(ims_names));
for l=1:length(ims_names)
    im = imread(strcat(IMAGES,'/',ims_names(l).name));
    im = imresize(im, [row col]);
    im = im2double(im);
	video(:,:,l) = im(:,:);
end

% Remove noise by only considering the top percentage brightest moments of each px in the video
quota_20 = round(0.2*length(ims_names));
quota_5 = round(0.05*length(ims_names));

suma = zeros(row, col);
suma_20 = zeros(row, col);
suma_5 = zeros(row, col);

for i=1:row
    for j=1:col
        [sortedPx, sortIdx] = sort(video(i,j,:), 'descend');
        %Yield the sum of all moments of each pixel
        suma(i,j) = sum(sortedPx);    
        % Yields the sum of the 20% brightetst moments of each pixel
        s_20 = sortedPx(1:quota_20);
        suma_20(i,j) = sum(s_20);
        % Yields the sum of the 5% brightetst moments of each pixel
        s_5 = sortedPx(1:quota_5);
        suma_5(i,j) = sum(s_5);
    end
end

norm = max(suma(:)); % Normalizes the image
tot = suma/norm;

norm_20 = max(suma_20(:)); % Normalizes the image
tot_20 = suma_20/norm_20;

norm_5 = max(suma_5(:)); % Normalizes the image
tot_5 = suma_5/norm_5;

% Incorporates the 3 sums into a 3-channel image
new_image(:,:,1) = tot;
new_image(:,:,2) = tot_20;
new_image(:,:,3) = tot_5;

imwrite(new_image,strcat(ORIGIN,'/3-channel',dataSet,'.tiff'));  % Saves the 3-channel image


% Expands data and annotations by performing cropping according to a grid + rotations
    lim_row = row - cropSize; % Limiting sizes come from the image resizing
    lim_col = col - cropSize;
    
 	for i=1:100:lim_row
        for j=1:100:lim_col  
        
        im_cropped = new_image(i:i+cropSize, j:j+cropSize, :);
        ann_cropped = ann(i:i+cropSize, j:j+cropSize);
    
            for deg=0:180:181
            im_cropped_r = imrotate(im_cropped,deg);
            ann_cropped_r = imrotate(ann_cropped,deg);
        
            %cd(CROPS)
            formatSpec = '%s/im_%s_%d-%d_%d.jpg';
            filename = sprintf(formatSpec,CROPS,dataSet,i,j,deg);
            imwrite(im_cropped_r,filename);
        
            %cd(ANNOTATIONS)
            formatSpec = '%s/ann_%s_%d-%d_%d.jpg';
            filename = sprintf(formatSpec,ANNOTATIONS,dataSet,i,j,deg);
            imwrite(ann_cropped_r,filename);
        
            end
        end
        
    end
