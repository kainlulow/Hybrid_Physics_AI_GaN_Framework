%% AlN dataset launcher for processDataset
clear; close all; clc;

csvPath = 'AlN.csv';

% Device id -> tag mapping (example labels from original script)
% 24463 -> AlN1-1, 24580 -> AlN1-0
try
    tagMap = containers.Map({'24463','24580'}, {'AlN1-1','AlN1-0'});
catch
    tagMap = struct('x24463','AlN1-1','x24580','AlN1-0'); %#ok<NASGU>
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

disp('AlN dataset processing complete.');
