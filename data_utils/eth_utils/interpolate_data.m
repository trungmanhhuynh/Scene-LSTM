function interpolate_data
 %% function to interpolate data for every frames.

 inputData = dlmread('original_data_meters.txt'); 

% Extract input data file to list of trajectories
 trajList = extract_trajectory(inputData);

% Linear interpolate each of trajectory list  
 interpolatedTrajList = interpolate_trajectory(trajList); 
  
% Convert trajectory list into annotation file format
%(frameId, targetID, x, y) 
 result = convert_trajectory_to_annotation(interpolatedTrajList,min(inputData(:,1)), max(inputData(:,1)));

 
 csvwrite('interpolated_data_meters.txt', result);

end 

%% Extract trajectory of each target and stored in trajList
function trajList = extract_trajectory(inputData)

targetIDList = unique(inputData(:,2)) ;
noTarget = length(unique(inputData(:,2))) ;    % number of target (i.e trajectories)
trajList = cell(1,noTarget) ;                  % each trajectory will be stored in
% a cell of trajList

for i = 1:1:noTarget
    trajList{i}.targetID = targetIDList(i) ;
    trajList{i}.frameList = inputData(inputData(:,2) == targetIDList(i),1) ;
    trajList{i}.locations = inputData(inputData(:,2) == targetIDList(i),[3 4]) ;
end

end

%% Interpolate each trajectory 
function result = interpolate_trajectory(trajList)
 
 result = cell(1, size(trajList,2)); 
% Process each trajectory 
 for i=1:1:size(trajList,2)
     
      result{i}.targetID = trajList{i}.targetID ;
      result{i}.frameList = [trajList{i}.frameList(1):trajList{i}.frameList(end)]';
      result{i}.locations = zeros(size(result{i}.frameList,1),2); 
      for frameId = result{i}.frameList(1):result{i}.frameList(end) 
          match_id = find(trajList{i}.frameList == frameId); 
          if(~isempty(match_id))
             %if this frameID is already existed then just copy the
             %original one
             result{i}.locations(result{i}.frameList == frameId,:) = trajList{i}.locations(match_id,:); 
          else
              %if not exist, then need to be interpolated using the closed
              %previous frame and next frame
              preFrame_id= max(find(trajList{i}.frameList < frameId)) ; 
              nextFrame_id = min(find(trajList{i}.frameList > frameId)) ;
              preFrame = trajList{i}.frameList(preFrame_id); 
              nextFrame = trajList{i}.frameList(nextFrame_id); 
              result{i}.locations(result{i}.frameList == frameId,:) = ...
              trajList{i}.locations(preFrame_id,:)+(frameId - preFrame)/(nextFrame-preFrame)...
              *(trajList{i}.locations(nextFrame_id,:) - trajList{i}.locations(preFrame_id,:)); 
          end
      end 
      % Floor all locations data to get integer values
      %result{i}.locations = floor(result{i}.locations); 
 end 

end 

%% Convert trajectory list back to annotation data file of format
% (frameId, targetID, x, y) 
function result = convert_trajectory_to_annotation(TrajList,minFrameId, maxFrameId)

 result = [] ;
 for frameId = minFrameId:1:maxFrameId
     %Scan through each trajectory to see if it has data in current frameID
     for i = 1:size(TrajList,2)
         match_id = find(TrajList{i}.frameList == frameId) ; 
          if(~isempty(match_id))
              % Build a row for this target 
                aRow = [frameId , TrajList{i}.targetID,  TrajList{i}.locations(match_id,:)] ;
                result = [result ; aRow ] ;
          end 
     end 
     frameId 
 end 

end 

