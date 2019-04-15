%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Description:
%  Author: Huynh
%  Date: 12/11/2018
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function convert_world_origin_meters
    
    % Read transformation matrix given by the dataset
    H = dlmread('H.txt');
    
    % Convert origin [0,0,1] (right-top corner of image) to real-world coordinate.
    origin_world = H*[0,0,1]'
    origin_world(1) = origin_world(1)/origin_world(3)
    origin_world(2) = origin_world(2)/origin_world(3)

    % Get translation matrix in real-world coordniate
    T = -[origin_world(1) , origin_world(2)];
    
    % Get new world data 
    input_data = dlmread('original_interpolated_data_meters.txt');
    output_data = input_data ;
    output_data(:,3) = input_data(:,3) + T(1);
    output_data(:,4) = input_data(:,4) + T(2);
    
    
    temp = output_data(:,3) ; 
    output_data(:,3) = output_data(:,4)  ;
    output_data(:,4) = temp ;

    
    csvwrite('data_meters.txt', output_data);

    fprintf("done\n")

end 