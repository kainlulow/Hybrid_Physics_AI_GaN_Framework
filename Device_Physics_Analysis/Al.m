%% Al dataset launcher for processDataset
clear; close all; clc;

csvPath = 'Al.csv';

% Device id -> tag mapping
% 24352 -> Low-Al, 24463 -> High-Al
try
    tagMap = containers.Map({'24352','24463'}, {'Low-Al','High-Al'});
catch
    % Fallback for older MATLAB without containers.Map in path
    tagMap = struct('x24352','Low-Al','x24463','High-Al'); %#ok<NASGU>
end

% Options
options = struct();
options.makePlots      = true;
options.extrapMode     = 'extrap';
options.saveSummary    = true;
options.W_cm           = 1;                  % width (cm)
options.runLabel       = datestr(now,'yyyymmdd_HHMMSS');
options.visibleFigures = 'off';

% Execute unified pipeline
outputs = processDataset(csvPath, tagMap, options); %#ok<NASGU>

disp('Al dataset processing complete.');
