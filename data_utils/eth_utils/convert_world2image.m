function convert_world2image()
%% function to convert world coordinates (in meters) to image coordinate (in pixels)
% used for ETH dataset (ETH_Hotel, ETH_Univ)
%%
H = dlmread('..\..\raw_data\eth_hotel\H.txt');                                 % read homography matrix 
world_data = dlmread('..\..\raw_data\eth_hotel\original_interpolated_meters.txt');     % read world data (in meters)
pos = [world_data(:,3) world_data(:,4) ones(size(world_data,1),1)];         % homogeneous coordinates [z ,y , 1]
pixels_unnormalized = pinv(H) * pos';                                       % Assuming H converts world coordinates 
                                                                            % to image coordinates
pixels_normalized = bsxfun(@rdivide, pixels_unnormalized([1,2],:), ...      # Divided the third col
                              pixels_unnormalized(3,:));
pixels_data = [world_data(:,1)'; world_data(:,2)';pixels_normalized];

% pixel_pos_frame_ped 3rd row is x, 4th row is y
temp = pixels_data(3,:) ;                                                   % For ETH, we need to switch x and y
pixels_data(3,:) = pixels_data(4,:)  ;                                     
pixels_data(4,:) = temp ;

% Save the positions to a mat file
csvwrite('original_interpolated_pixels.txt', pixels_data');

end 