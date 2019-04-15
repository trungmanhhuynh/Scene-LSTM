function normalize_data
%% function to normalize data (pixels or meters) to range [0,1] or [-1,1]
% Author : Huynh Manh

%% Required parameters
use_pixel = 1  ;                                                            % 1 if data in pixels
                                                                            % 0 if data in meters
range = 1 ;                                                                 % 1 if convert to range [-1, 1]
                                                                            % 0 if convert to range [0, 1]
inputData = dlmread('..\..\raw_data\eth_univ\data_pixels_2.5fps.txt');% Specify data file
if(use_pixel)
    maxWidth  = 640; minWidth = 0 ;                                         % ETH_Hotel: [w,h] = [720,576]
    maxHeight = 480; minHeight = 0 ;                                        % ETH_Univ:  [w,h] = [640,480]
else
    maxWidth  = max(inputData(:,3)) ;  minWidth  = min(inputData(:,3)) ;
    maxHeight  = max(inputData(:,4)) ;  minHeight  = min(inputData(:,4)) ;
end

%% 
normalizedData = inputData ;
if(range == 0)
    disp("Normalize (x,y) locations into range [0,1]")
    normalizedData(:,3) = normalizedData(:,3)/frameWidth ; % normalize x
    normalizedData(:,4) = normalizedData(:,4)/frameHeight ; % normalize y
elseif(range == 1) 
    disp("Normalize (x,y) locations into range [-1,1]")
    normalizedData(:,3) = 2*(normalizedData(:,3) - minWidth)/(maxWidth - minWidth) -1 ; % normalize x
    normalizedData(:,4) = 2*(normalizedData(:,4) - minHeight)/(maxHeight - minHeight) -1 ; % normalize y
else 
   error('mode can be either 0 or 1'); 
end 
% Write file 
csvwrite('data_normalized_pixels_2.5fps.txt', normalizedData);
disp("Done")     

end 