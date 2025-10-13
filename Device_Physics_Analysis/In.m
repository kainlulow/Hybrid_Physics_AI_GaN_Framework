%% In dataset launcher for processDataset
clear; close all; clc;

csvPath = 'In.csv';

% Device id -> tag mapping (example labels from original script)
% 24352 -> Low-In, 24355 -> High-In
try
    tagMap = containers.Map({'24352','24355'}, {'Low-In','High-In'});
catch
    tagMap = struct('x24352','Low-In','x24355','High-In'); %#ok<NASGU>
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

disp('In dataset processing complete.');
