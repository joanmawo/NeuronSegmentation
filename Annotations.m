% Script that extracts the annotations from the JSON files.
% It produces two annotated images: one containing the contours of the neurons
% and a second one which distinguishes among different cell instances.
% This script makes use of the function LOADJSON(), which is part of the JSONLAB toolbox, available
% https://www.mathworks.com/matlabcentral/fileexchange/33381-jsonlab--a-toolbox-to-encode-decode-json-files.
function output = Annotations(dataSet)

ORIGIN = strcat('../', dataSet);
Json = strcat(ORIGIN,'/regions');
IMAGES = strcat(ORIGIN,'/images');
ANNOTATIONS = strcat('Annotations');

anns = loadjson(strcat(Json,'/regions.json'));
[ann_number1 ann_number2] = size(anns);

im_0 = imread(strcat(IMAGES, '/image00000.tiff'));
[dims1 dims2] = size(im_0);

contours = zeros(dims1, dims2);
instances = zeros(dims1, dims2);

for l=1:ann_number2

	[size1 size2] = size(anns{l}.coordinates);
	for j=1:size1
		
			contours(anns{l}.coordinates(j,1),anns{l}.coordinates(j,2)) = 1;
			instances(anns{l}.coordinates(j,1),anns{l}.coordinates(j,2)) = l;

	end 
end	

imwrite(contours,strcat(ORIGIN,'/contours',dataSet,'.tiff'));
imwrite(instances,strcat(ORIGIN,'/instances',dataSet,'.tiff'));

output = l;
