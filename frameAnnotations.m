% Script that produces frame-by-frame annotations for a NeuroFinder dataset video.
% The script compares the annotations to each frame. A neuron is said to be 'on' if the sum of
% its pixels' intesities surpasses a given threshold.
% A new annotation is stored for each frame.

function output = frameAnnotations(dataSet)

threshold = 0.1;


ORIGIN = strcat('neurofinder.',dataSet);
IMAGES = strcat(ORIGIN,'/images');
CROPS = strcat('crops/images');
ANNOTATIONS = strcat('crops/annotations');

regions =  imread(strcat(ORIGIN, '/contours', dataSet, '.tiff'));

load(strcat(ORIGIN, '/coordinates', dataSet, '.mat'));

[ann_number1 ann_number2] = size(anns);

ims_names = dir(strcat(IMAGES,'/*.tiff'));

im = imread(strcat(IMAGES,'/',ims_names(1).name));
[row col] = size(im);

frame_ann = zeros(row, col);

for k=1:length(ims_names)
    im = imread(strcat(IMAGES,'/',ims_names(k).name));
    
    suma = 0;

	for l=1:ann_number2

		[size1 size2] = size(anns{l}.coordinates);
		for j=1:size1
		
			suma = suma + im(anns{l}.coordinates(j,1),anns{l}.coordinates(j,2));
		end 

		suma = suma/size1;
	
		if suma > threshold	
			for j=1:size1
				frame_ann(anns{l}.coordinates(j,1),anns{l}.coordinates(j,2)) = 1;
			end 

		end

	end

end	
