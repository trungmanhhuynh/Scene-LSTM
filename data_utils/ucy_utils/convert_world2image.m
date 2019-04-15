function convert_world2image()
%% function to convert world coordinates (in meters) to image coordinate (in pixels)
% used for ETH dataset (ETH_Hotel, ETH_Univ)
%%
H = dlmread('..\..\raw_data\ucy_zara01\H.txt');                                 % read homography matrix 
world_data = dlmread('..\..\raw_data\ucy_zara01\data_meters_2.5fps.txt');     % read world data (in meters)
pos = [world_data(:,3) world_data(:,4) ones(size(world_data,1),1)];         % homogeneous coordinates [z ,y , 1]
pixels_unnormalized = pinv(H) * pos';                                       % Assuming H converts world coordinates 
                                                                            % to image coordinates
pixels_normalized = bsxfun(@rdivide, pixels_unnormalized([1,2],:), ...      # Divided the third col
                              pixels_unnormalized(3,:));
pixels_data = [world_data(:,1)'; world_data(:,2)';pixels_normalized];

% Save the positions to a mat file
csvwrite('data_pixels.txt', pixels_data');

end 