function outputs = processDataset(csvPath, tagMap, options)
% processDataset: Unified pipeline for Device_Physics_Analysis datasets.
% Inputs
%  - csvPath: path to dataset CSV (headers preserved as exported)
%  - tagMap: mapping from device id string -> devTag (containers.Map or struct)
%  - options: struct with fields (all optional):
%       makePlots      (logical, default true)
%       extrapMode     ('extrap' or NaN, default 'extrap')
%       saveSummary    (logical, default true)
%       W_cm           (double width in cm, default 1)
%       runLabel       (char, default datestr now)
%       resDir         (char, default <csv>_processed_<runLabel>)
%       plotResolution (integer, default 200)
%       visibleFigures ('on'|'off', default 'off')
%
% Outputs (struct)
%   - resDir, allCSV, sumCSV, tradeoffCSV, tradeoffDeviceCSV

if nargin < 3 || isempty(options), options = struct(); end

% ---- defaults
defaults                 = struct();
defaults.makePlots       = true;
defaults.extrapMode      = 'extrap';
defaults.saveSummary     = true;
defaults.W_cm            = 1;
defaults.runLabel        = datestr(now,'yyyymmdd_HHMMSS');
defaults.resDir          = '';
defaults.plotResolution  = 200;
defaults.visibleFigures  = 'off';

opt = mergeOptions(defaults, options);

% ---- Read input CSV
Traw = readtable(csvPath, 'PreserveVariableNames', true);
vars = Traw.Properties.VariableNames;

% Regex for headers
pat = '^(hDensity|hVelocity|hMobility|hSaturationVelocity|ElectricField-X)\((.*)\)\s*([XY])$';

entries_prop = {}; entries_key = {}; entries_axis = {}; entries_col = [];
for i = 1:numel(vars)
    v = vars{i};
    tok = regexp(v, pat, 'tokens', 'once');
    if ~isempty(tok)
        entries_prop{end+1,1} = tok{1}; %#ok<AGROW>
        entries_key{end+1,1}  = tok{2}; %#ok<AGROW>
        entries_axis{end+1,1} = tok{3}; %#ok<AGROW>
        entries_col(end+1,1)  = i;      %#ok<AGROW>
    end
end

% Map: key -> indices for each property/axis
keyNamesRaw = unique(entries_key);
keyMap = struct();
for k = 1:numel(keyNamesRaw)
    keyRaw = keyNamesRaw{k};
    kk = matlab.lang.makeValidName(keyRaw);
    keyMap.(kk) = struct('rawKey', keyRaw, ...
        'dX', [], 'dY', [], 'vX', [], 'vY', [], 'mX', [], 'mY', [], ...
        'sX', [], 'sY', [], 'eX', [], 'eY', []);
end

for i = 1:numel(entries_col)
    prop  = entries_prop{i};
    keyRaw = entries_key{i};
    kk    = matlab.lang.makeValidName(keyRaw);
    ax    = entries_axis{i};
    col   = entries_col(i);
    switch prop
        case 'hDensity'
            if ax == "X", keyMap.(kk).dX = col; else, keyMap.(kk).dY = col; end
        case 'hVelocity'
            if ax == "X", keyMap.(kk).vX = col; else, keyMap.(kk).vY = col; end
        case 'hMobility'
            if ax == "X", keyMap.(kk).mX = col; else, keyMap.(kk).mY = col; end
        case 'hSaturationVelocity'
            if ax == "X", keyMap.(kk).sX = col; else, keyMap.(kk).sY = col; end
        case 'ElectricField-X'
            if ax == "X", keyMap.(kk).eX = col; else, keyMap.(kk).eY = col; end
    end
end

% Keep only samples that have Density, Velocity, SatVelocity, Ex (both X and Y)
allK = fieldnames(keyMap);
keep = false(size(allK));
for i = 1:numel(allK)
    K = keyMap.(allK{i});
    keep(i) = ~isempty(K.dX) && ~isempty(K.dY) && ...
              ~isempty(K.vX) && ~isempty(K.vY) && ...
              ~isempty(K.sX) && ~isempty(K.sY) && ...
              ~isempty(K.eX) && ~isempty(K.eY);
end
allK = allK(keep);

if isempty(allK)
    fprintf('Found 0 samples with required columns. Check headers. Example columns:\n');
    disp(vars(1:min(12,numel(vars))));
    warning('No output rows were generated.');
    outputs = struct('resDir','');
    return;
end

fprintf('Found %d samples:\n', numel(allK));
for i = 1:numel(allK)
    fprintf('  %s\n', keyMap.(allK{i}).rawKey);
end

% ---- Prepare output directory
[outDir, baseName, ~] = fileparts(csvPath);
if isempty(outDir), outDir = '.'; end
if isempty(opt.resDir)
    resDir = fullfile(outDir, sprintf('%s_processed_%s', baseName, opt.runLabel));
else
    resDir = opt.resDir;
end
if ~exist(resDir, 'dir'), mkdir(resDir); end

% Combined outputs
outAll = table();
summaryRows = table('Size',[0 8], ...
    'VariableTypes', {'string','double','double','double','double','double','double','double'}, ...
    'VariableNames', {'sampleKey','Integral_pv_over_x_cm1s','Integral_pvSat_over_x_cm1s','Integral_pmuEx_over_x_cm1s', ...
                      'Integral_hdensity_over_x_cm2','Integral_hvelocity_over_x_cms','Integral_hmobility_over_x_cm2Vs','Integral_hSaturationVelocity_over_x_cms'});

% ---- Process each sample
for i = 1:numel(allK)
    K = keyMap.(allK{i});
    keyRaw  = K.rawKey;
    safeKey = regexprep(keyRaw, '[^\w\-]+', '_');

    % Extract series
    x_d = Traw{:, K.dX};   p  = Traw{:, K.dY};   % hDensity vs x (um), p (cm^-3)
    x_v = Traw{:, K.vX};   v  = Traw{:, K.vY};   % hVelocity vs x (um), v (cm/s)
    x_s = Traw{:, K.sX};   vs = Traw{:, K.sY};   % v_sat vs x (um), vs (cm/s)
    x_e = Traw{:, K.eX};   ex = Traw{:, K.eY};   % Ex vs x (um), Ex (V/cm)

    mu = []; x_mu = [];
    if ~isempty(K.mX) && ~isempty(K.mY)
        x_mu = Traw{:, K.mX};      % um
        mu   = Traw{:, K.mY};      % cm^2/(V·s)
    end

    % Clean NaNs
    valid_p = isfinite(x_d) & isfinite(p); x_d = x_d(valid_p); p = p(valid_p);
    valid_v = isfinite(x_v) & isfinite(v); x_v = x_v(valid_v); v = v(valid_v);
    valid_s = isfinite(x_s) & isfinite(vs); x_s = x_s(valid_s); vs = vs(valid_s);
    valid_e = isfinite(x_e) & isfinite(ex); x_e = x_e(valid_e); ex = ex(valid_e);

    if isempty(x_d) || isempty(x_v) || isempty(x_s) || isempty(x_e)
        warning('Skipping "%s": empty data after cleaning.', keyRaw);
        continue;
    end

    % Sort by position
    [x_d, iD] = sort(x_d); p = p(iD);
    [x_v, iV] = sort(x_v); v = v(iV);
    [x_s, iS] = sort(x_s); vs = vs(iS);
    [x_e, iE] = sort(x_e); ex = ex(iE);

    % Deduplicate for non-density series: keep last entry per position bin
    tolV = max(1e-12, 10*eps(max(1, abs(x_v(end)))));
    binsV = round(x_v / tolV);
    [~, lastIdx] = unique(binsV, 'last');
    x_v = x_v(lastIdx);  v  = v(lastIdx);

    tolS = max(1e-12, 10*eps(max(1, abs(x_s(end)))));
    binsS = round(x_s / tolS);
    [~, lastIdxS] = unique(binsS, 'last');
    x_s = x_s(lastIdxS); vs = vs(lastIdxS);

    tolE = max(1e-12, 10*eps(max(1, abs(x_e(end)))));
    binsE = round(x_e / tolE);
    [~, lastIdxE] = unique(binsE, 'last');
    x_e = x_e(lastIdxE); ex = ex(lastIdxE);

    if ~isempty(mu)
        valid_mu = isfinite(x_mu) & isfinite(mu);
        x_mu = x_mu(valid_mu); mu = mu(valid_mu);
        [x_mu, iM] = sort(x_mu); mu = mu(iM);
        tolM = max(1e-12, 10*eps(max(1, abs(x_mu(end)))));
        binsM = round(x_mu / tolM);
        [~, lastIdxM] = unique(binsM, 'last');
        x_mu = x_mu(lastIdxM); mu = mu(lastIdxM);
    end

    % Units: um -> cm for all x arrays
    um_to_cm = 1e-4;
    x_d = x_d * um_to_cm;
    x_v = x_v * um_to_cm;
    x_s = x_s * um_to_cm;
    x_e = x_e * um_to_cm;
    if ~isempty(x_mu), x_mu = x_mu * um_to_cm; end

    % Interpolate onto density grid (x_d [cm])
    v_on_xd  = interp1(x_v, v,  x_d, 'linear', opt.extrapMode);
    vs_on_xd = interp1(x_s, vs, x_d, 'linear', opt.extrapMode);
    Ex_on_xd = interp1(x_e, ex, x_d, 'linear', opt.extrapMode);
    if ~isempty(mu)
        mu_on_xd = interp1(x_mu, mu, x_d, 'linear', opt.extrapMode);
    else
        mu_on_xd = nan(size(x_d));
    end

    % Products (cm^-2 s^-1)
    pv    = p .* v_on_xd;
    pvSat = p .* vs_on_xd;
    pMuEx = p .* mu_on_xd .* Ex_on_xd;

    % Running integrals (cm^-1 s^-1)
    if numel(x_d) >= 2
        dx = diff(x_d);
        avg_pv    = 0.5 * (pv(1:end-1)    + pv(2:end));
        avg_pvSat = 0.5 * (pvSat(1:end-1) + pvSat(2:end));
        avg_pMuEx = 0.5 * (pMuEx(1:end-1) + pMuEx(2:end));
        cumFlux        = [0; cumsum(dx .* avg_pv)];
        cumFluxSat     = [0; cumsum(dx .* avg_pvSat)];
        cumFlux_pMuEx  = [0; cumsum(dx .* avg_pMuEx)];
    else
        cumFlux = zeros(size(x_d));
        cumFluxSat = zeros(size(x_d));
        cumFlux_pMuEx = zeros(size(x_d));
    end

    % Total integrals over x (cm^-1 s^-1)
    integral_pv    = trapz(x_d, pv);
    integral_pvSat = trapz(x_d, pvSat);
    integral_pmuEx = trapz(x_d, pMuEx);

    % Individual field integrals
    integral_hdensity_over_x            = trapz(x_d, p);        % cm^-2
    integral_hvelocity_over_x           = trapz(x_d, v_on_xd);  % cm^2/s
    integral_hmobility_over_x           = trapz(x_d, mu_on_xd); % cm^3/(V·s)
    integral_hSaturationVelocity_over_x = trapz(x_d, vs_on_xd); % cm^2/s

    % Per-sample table
    Ts = table(x_d, p, v_on_xd, mu_on_xd, pv, cumFlux, repmat(integral_pv, size(x_d)), ...
               vs_on_xd, pvSat, cumFluxSat, repmat(integral_pvSat, size(x_d)), ...
               Ex_on_xd, pMuEx, cumFlux_pMuEx, repmat(integral_pmuEx, size(x_d)), ...
        'VariableNames', {'x_cm','hDensity_cm3','hVelocity_cms_interp','hMobility_cm2Vs_interp', ...
                          'hDensity_times_hVelocity_cm2s','CumulativeIntegral_pv_cm1s','Integral_pv_over_x_cm1s', ...
                          'hSatVelocity_cms_interp','hDensity_times_hSatVelocity_cm2s','CumulativeIntegral_pvSat_cm1s','Integral_pvSat_over_x_cm1s', ...
                          'Ex_Vpcm_interp','hDensity_times_hMobility_times_Ex_cm2s','CumulativeIntegral_pmuEx_cm1s','Integral_pmuEx_over_x_cm1s'});

    % Save per-sample CSV
    outCSV = fullfile(resDir, sprintf('sample_%s.csv', safeKey));
    writetable(Ts, outCSV);

    % Accumulate combined table (all three products)
    Ts.sampleKey = repmat({keyRaw}, height(Ts), 1);
    outAll = [outAll; Ts]; %#ok<AGROW>

    % Accumulate summary row
    summaryRows = [summaryRows; { ...
        string(keyRaw), ...
        integral_pv, ...
        integral_pvSat, ...
        integral_pmuEx, ...
        integral_hdensity_over_x, ...
        integral_hvelocity_over_x, ...
        integral_hmobility_over_x, ...
        integral_hSaturationVelocity_over_x ...
    }]; %#ok<AGROW>

    % Per-sample plots
    if opt.makePlots
        f = figure('Visible',opt.visibleFigures,'Color','w','Name',keyRaw);
        t = tiledlayout(f,2,4,'TileSpacing','compact','Padding','compact');

        nexttile; plot(x_d, p, '-'); grid on;
        xlabel('x (cm)'); ylabel('hDensity (cm^{-3})'); title(sprintf('Density — %s', keyRaw), 'Interpreter','none');

        nexttile; plot(x_d, v_on_xd, '-'); grid on;
        xlabel('x (cm)'); ylabel('hVelocity (cm/s)'); title('Velocity on density grid');

        nexttile; plot(x_d, vs_on_xd, '-'); grid on;
        xlabel('x (cm)'); ylabel('hSatVelocity (cm/s)'); title('Sat. velocity on density grid');

        nexttile; plot(x_d, Ex_on_xd, '-'); grid on;
        xlabel('x (cm)'); ylabel('Ex (V/cm)'); title('Lateral Electric Field Ex');

        nexttile; plot(x_d, pv, '-'); grid on;
        xlabel('x (cm)'); ylabel('p \times v (cm^{-2} s^{-1})'); title('Product (p \times v)');

        nexttile; plot(x_d, pvSat, '-'); grid on;
        xlabel('x (cm)'); ylabel('p \times v_{sat} (cm^{-2} s^{-1})'); title('Product (p \times v_{sat})');

        nexttile; plot(x_d, pMuEx, '-'); grid on;
        xlabel('x (cm)'); ylabel('p \times \mu \times Ex (cm^{-2} s^{-1})'); title('Product (p \times \mu \times Ex)');

        nexttile;
        if any(isfinite(mu_on_xd))
            plot(x_d, mu_on_xd, '-'); grid on;
            xlabel('x (cm)'); ylabel('\mu (cm^{2}/(V\cdot s))'); title('Mobility');
        else
            axis off; text(0.5,0.5,'Mobility N/A','HorizontalAlignment','center');
        end

        outPNG = fullfile(resDir, sprintf('sample_%s.png', safeKey));
        exportgraphics(t, outPNG, 'Resolution', opt.plotResolution);
        close(f);

        ratio = integral_pv / integral_pmuEx;  % ~1 if v ≈ μ Ex (linear regime)
        fprintf('Check [%s]: ∫p v dx / ∫p μ Ex dx = %.3f\n', keyRaw, ratio);
    end
end

% Combined product summary plot (all three products)
if opt.makePlots && ~isempty(outAll)
    f2 = figure('Visible',opt.visibleFigures,'Color','w','Name','All Samples — Products');
    tl = tiledlayout(f2,3,1,'Padding','compact','TileSpacing','compact');
    keysList = unique(outAll.sampleKey, 'stable');

    ax1 = nexttile; hold(ax1,'on'); grid(ax1,'on');
    for kk = 1:numel(keysList)
        mask = strcmp(outAll.sampleKey, keysList{kk});
        plot(outAll.x_cm(mask), outAll.hDensity_times_hVelocity_cm2s(mask), '-');
    end
    xlabel('x (cm)'); ylabel('p \times v (cm^{-2} s^{-1})'); title('All Samples — Product (p \times v)');
    legend(keysList, 'Interpreter','none', 'Location','bestoutside');

    ax2 = nexttile; hold(ax2,'on'); grid(ax2,'on');
    for kk = 1:numel(keysList)
        mask = strcmp(outAll.sampleKey, keysList{kk});
        plot(outAll.x_cm(mask), outAll.hDensity_times_hSatVelocity_cm2s(mask), '-');
    end
    xlabel('x (cm)'); ylabel('p \times v_{sat} (cm^{-2} s^{-1})'); title('All Samples — Product (p \times v_{sat})');

    ax3 = nexttile; hold(ax3,'on'); grid(ax3,'on');
    for kk = 1:numel(keysList)
        mask = strcmp(outAll.sampleKey, keysList{kk});
        plot(outAll.x_cm(mask), outAll.hDensity_times_hMobility_times_Ex_cm2s(mask), '-');
    end
    xlabel('x (cm)'); ylabel('p \times \mu \times Ex (cm^{-2} s^{-1})'); title('All Samples — Product (p \times \mu \times Ex)');

    sumPNG = fullfile(resDir, 'all_samples_products.png');
    exportgraphics(tl, sumPNG, 'Resolution', opt.plotResolution);
    close(f2);
end

% Save combined and summary CSVs
allCSV = fullfile(resDir, 'all_samples_density_velocity_product.csv');
if ~isempty(outAll)
    writetable(outAll, allCSV);
    fprintf('Wrote combined results to: %s\n', allCSV);
else
    warning('No output rows were generated for the combined file.');
end

sumCSV = '';
if opt.saveSummary && ~isempty(summaryRows)
    sumCSV = fullfile(resDir, 'summary_integrals.csv');
    writetable(summaryRows, sumCSV);
    fprintf('Wrote per-sample integral summary to: %s\n', sumCSV);
end

% ---- Post-processing from the summary file ----
tradeoffCSV = '';
tradeoffDeviceCSV = '';
if ~isempty(sumCSV) && isfile(sumCSV)
    q    = 1.602176634e-19;  % C
    W_cm = opt.W_cm;         % width (cm)

    TSUM = readtable(sumCSV, 'PreserveVariableNames', true);
    if height(TSUM) > 0
        % Parse device id and step index
        nRow = height(TSUM);
        devID  = strings(nRow,1);
        stepID = NaN(nRow,1);
        for i = 1:nRow
            s = string(TSUM.sampleKey{i});
            tok = regexp(s, 'n[_-]*([0-9]+)', 'tokens', 'once');
            if ~isempty(tok), devID(i) = string(tok{1}); else, devID(i) = string(missing); end
            tok2 = regexp(s, 'inter[_-]*([0-9]+)', 'tokens', 'once');
            if ~isempty(tok2), stepID(i) = str2double(tok2{1}); else, stepID(i) = NaN; end
        end
        TSUM.devID  = devID;
        TSUM.stepID = stepID;

        % Drop unparsable rows
        badMask = isnan(TSUM.stepID) | ismissing(TSUM.devID);
        if any(badMask)
            warning('Skipping %d rows with unparsable sampleKey:', nnz(badMask));
            disp(TSUM.sampleKey(badMask));
            TSUM(badMask,:) = [];
        end

        % Device tagging via tagMap
        TSUM.devTag = repmat("Unknown", height(TSUM), 1);
        if isa(tagMap, 'containers.Map')
            ids = unique(TSUM.devID(~ismissing(TSUM.devID)));
            for i = 1:numel(ids)
                idc = char(ids(i));
                if isKey(tagMap, idc)
                    TSUM.devTag(TSUM.devID==ids(i)) = string(tagMap(idc));
                end
            end
        elseif isstruct(tagMap)
            ids = unique(TSUM.devID(~ismissing(TSUM.devID)));
            fns = fieldnames(tagMap);
            for i = 1:numel(ids)
                idc = char(ids(i));
                if any(strcmp(fns, idc))
                    TSUM.devTag(TSUM.devID==ids(i)) = string(tagMap.(idc));
                end
            end
        end

        % Core integrals and derived metrics
        I_pv_int_cm1s    = TSUM.Integral_pv_over_x_cm1s;
        I_pvSat_int_cm1s = TSUM.Integral_pvSat_over_x_cm1s;
        I_pmuEx_int_cm1s = TSUM.Integral_pmuEx_over_x_cm1s;
        Qp_int_cm2       = TSUM.Integral_hdensity_over_x_cm2;

        TSUM.Qp_cm2         = Qp_int_cm2;
        TSUM.v_avg_p_cms    = I_pv_int_cm1s    ./ Qp_int_cm2;   % cm/s
        TSUM.muEx_avg_p_cms = I_pmuEx_int_cm1s ./ Qp_int_cm2;   % cm/s
        TSUM.vsat_avg_p_cms = I_pvSat_int_cm1s ./ Qp_int_cm2;   % cm/s
        TSUM.eta_v          = I_pv_int_cm1s ./ I_pvSat_int_cm1s;
        TSUM.r_lin          = I_pv_int_cm1s ./ I_pmuEx_int_cm1s;

        % Current per width (A)
        TSUM.I_pv_AperW = q * W_cm * I_pv_int_cm1s;

        TSUM = sortrows(TSUM, {'devTag','stepID'});

        % Dynamic color map from devTag
        devs = cellstr(unique(TSUM.devTag, 'stable'));
        pal   = lines(max(1, numel(devs)));
        colors = containers.Map();
        for i = 1:numel(devs)
            colors(devs{i}) = pal(i,:);
        end

        % ----- Figure: metrics vs gate step -----
        if opt.makePlots && ~isempty(devs)
            f2 = figure('Visible',opt.visibleFigures,'Color','w','Name','Metrics vs gate step');
            tl = tiledlayout(f2,3,2,'TileSpacing','compact','Padding','compact');

            % Qp (cm^-2)
            nexttile; hold on; grid on;
            for i = 1:numel(devs)
                k = devs{i};
                mask = (TSUM.devTag == string(k));
                [steps, ord] = sort(TSUM.stepID(mask));
                y = TSUM.Qp_cm2(mask); y = y(ord);
                plot(steps, y, '-o', 'DisplayName', k, 'Color', colors(k));
            end
            title('Q_p vs step'); xlabel('step'); ylabel('Q_p (cm^{-2})'); legend;

            % <v>_p (cm/s)
            nexttile; hold on; grid on;
            for i = 1:numel(devs)
                k = devs{i};
                mask = (TSUM.devTag == string(k)); [steps, ord] = sort(TSUM.stepID(mask));
                y = TSUM.v_avg_p_cms(mask); y = y(ord);
                plot(steps, y, '-o', 'DisplayName', k, 'Color', colors(k));
            end
            title('<v>_p vs step'); xlabel('step'); ylabel('<v>_p (cm/s)');

            % <μE>_p (cm/s)
            nexttile; hold on; grid on;
            for i = 1:numel(devs)
                k = devs{i};
                mask = (TSUM.devTag == string(k)); [steps, ord] = sort(TSUM.stepID(mask));
                y = TSUM.muEx_avg_p_cms(mask); y = y(ord);
                plot(steps, y, '-o', 'DisplayName', k, 'Color', colors(k));
            end
            title('<\mu E_x>_p vs step'); xlabel('step'); ylabel('<\mu E_x>_p (cm/s)');

            % <v_sat>_p (cm/s)
            nexttile; hold on; grid on;
            for i = 1:numel(devs)
                k = devs{i};
                mask = (TSUM.devTag == string(k)); [steps, ord] = sort(TSUM.stepID(mask));
                y = TSUM.vsat_avg_p_cms(mask); y = y(ord);
                plot(steps, y, '-o', 'DisplayName', k, 'Color', colors(k));
            end
            title('<v_{sat}>_p vs step'); xlabel('step'); ylabel('<v_{sat}>_p (cm/s)');

            % eta_v
            nexttile; hold on; grid on;
            for i = 1:numel(devs)
                k = devs{i};
                mask = (TSUM.devTag == string(k)); [steps, ord] = sort(TSUM.stepID(mask));
                y = TSUM.eta_v(mask); y = y(ord);
                plot(steps, y, '-o', 'DisplayName', k, 'Color', colors(k));
            end
            title('\eta_v = <v>_p / <v_{sat}>_p'); xlabel('step'); ylabel('\eta_v');

            % r = <v>_p / <μE>_p
            nexttile; hold on; grid on;
            for i = 1:numel(devs)
                k = devs{i};
                mask = (TSUM.devTag == string(k)); [steps, ord] = sort(TSUM.stepID(mask));
                y = TSUM.r_lin(mask); y = y(ord);
                plot(steps, y, '-o', 'DisplayName', k, 'Color', colors(k));
            end
            title('r = <v>_p / <\mu E_x>_p'); xlabel('step'); ylabel('r');

            exportgraphics(tl, fullfile(resDir,'metrics_vs_step.png'), 'Resolution', opt.plotResolution);
            close(f2);
        end

        % ----- Figure: Δln(I) decomposition -----
        if opt.makePlots && ~isempty(devs)
            f3 = figure('Visible',opt.visibleFigures,'Color','w','Name','Δln(I) decomposition');
            t3 = tiledlayout(f3, max(1,numel(devs)), 1, 'TileSpacing','compact','Padding','compact');

            Sum = table('Size',[0 8], ...
                'VariableTypes', {'string','double','double','double','double','double','double','double'}, ...
                'VariableNames', {'devTag','nTransitions','sum_dlnI','sum_dlnQ','sum_dlnv', ...
                                  'mean_shareQ','mean_shareV','final_cum_shareQ'});

            for i = 1:numel(devs)
                k  = devs{i};
                ax = nexttile; hold(ax,'on'); grid(ax,'on');

                mask = (TSUM.devTag == string(k));
                [stepsAll, ord] = sort(TSUM.stepID(mask));
                Q_all = TSUM.Qp_cm2(mask);         Q_all = Q_all(ord);
                v_all = TSUM.v_avg_p_cms(mask);    v_all = v_all(ord);
                I_all = TSUM.I_pv_AperW(mask);     I_all = I_all(ord);

                [G, stepU] = findgroups(stepsAll);
                Qg = splitapply(@mean, Q_all, G);
                vg = splitapply(@mean, v_all, G);
                Ig = splitapply(@mean, I_all, G);

                if numel(stepU) < 2
                    title(ax, sprintf('Decomposition — %s (insufficient steps)', k));
                else
                    dlnI = diff(log(Ig));
                    dlnQ = diff(log(Qg));
                    dlnv = diff(log(vg));

                    xk = 1:numel(dlnI);
                    B = bar(ax, xk, [dlnQ(:), dlnv(:)], 0.8, 'grouped');
                    B(1).FaceColor = [0.2 0.6 0.85];  B(1).DisplayName = '\Delta ln Q_p';
                    B(2).FaceColor = [0.85 0.4 0.2];  B(2).DisplayName = '\Delta ln <v>_p';

                    plot(ax, xk, dlnI, 'k.-', 'DisplayName','\Delta ln I (sum)');

                    xticks(ax, xk);
                    xticklabels(ax, compose('%d→%d', stepU(1:end-1), stepU(2:end)));
                    xlabel(ax, 'transition (step k \rightarrow k+1)');
                    ylabel(ax, '\Delta ln(\cdot)');
                    title(ax, sprintf('Decomposition — %s', k));
                    legend(ax,'Location','best');
                end

                % Device-level cumulative stats
                dlnI_all = [NaN; diff(log(I_all))];
                dlnQ_all = [NaN; diff(log(Q_all))];
                dlnv_all = [NaN; diff(log(v_all))];
                valid = find(~isnan(dlnI_all));
                nTr   = numel(valid);
                sI    = nansum(dlnI_all);
                sQ    = nansum(dlnQ_all);
                sv    = nansum(dlnv_all);
                meanQ = nanmean(dlnQ_all(valid) ./ dlnI_all(valid));
                meanV = nanmean(dlnv_all(valid) ./ dlnI_all(valid));
                finalCumShareQ = sQ / sI;

                Sum = [Sum; {string(k), nTr, sI, sQ, sv, meanQ, meanV, finalCumShareQ}]; %#ok<AGROW>
            end

            exportgraphics(t3, fullfile(resDir,'tradeoff_decomposition.png'), 'Resolution', opt.plotResolution);
            close(f3);

            % Save augmented per-step table and compact summary
            out1 = fullfile(resDir, 'summary_integrals_with_tradeoff.csv');
            writetable(TSUM, out1);
            fprintf('Wrote augmented per-step results to: %s\n', out1);

            out2 = fullfile(resDir, 'tradeoff_transition_summary.csv');
            writetable(Sum, out2);
            fprintf('Wrote device-level transition summary to: %s\n', out2);

            tradeoffCSV = out1; tradeoffDeviceCSV = out2;
        end
    end
end

% ---- outputs ----
outputs = struct('resDir', resDir, 'allCSV', allCSV, 'sumCSV', sumCSV, ...
                 'tradeoffCSV', tradeoffCSV, 'tradeoffDeviceCSV', tradeoffDeviceCSV);
end

% ===== helpers =====
function out = mergeOptions(defaults, opts)
    out = defaults;
    if isempty(opts), return; end
    f = fieldnames(opts);
    for i = 1:numel(f)
        out.(f{i}) = opts.(f{i});
    end
end
