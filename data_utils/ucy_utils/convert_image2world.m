function convert_image2world
%% function to convert image coordinates (pixels) to world coordinates (meters)
H = dlmread('..\..\raw_data\ucy_zara02\H.txt');                                                       % homography matrix
pixels_data = dlmread('..\..\raw_data\ucy_zara02\original_interpolated_pixels.txt');

% pos = [xpos ypos 1]
pos = [pixels_data(:,3) pixels_data(:,4) ones(size(pixels_data,1),1)];
meter_pos = H * pos';

% Normalize pixel pos by dividing by the third col
meter_pos = bsxfun(@rdivide, meter_pos([1,2],:), ...
                              meter_pos(3,:));

meter_pos = meter_pos';
meter_pos = [pixels_data(:,[1 2]) ,meter_pos];                              % concatenate [frame id, pid]

csvwrite('interpolated_data_meters.txt', meter_pos);
fprintf("done\n")

end 