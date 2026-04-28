function run_CI_cochlea(signalfilename_left, signalfilename_right, paramfilename, savepath, ci_model_root)

    addpath(genpath(ci_model_root));
    cd(fullfile(ci_model_root, 'GUI'));

    fs_AN = 10000;

    for side = {'left', 'right'}
        if strcmp(side{1}, 'left')
            signalfilename = signalfilename_left;
        else
            signalfilename = signalfilename_right;
        end

        % --- load params ---
        run(paramfilename);

        % --- read and normalize signal ---
        [signal, fsOrg] = audioread(signalfilename);
        signal = signal(:);
        signal = signal ./ sqrt(mean(signal.^2));
        signal = signal * 10^(-17/20);

        % --- CI signal processing ---
        [mINew, CIParams.pps, maxima] = ACE_signal_processing(signal, fsOrg, CIParams);
        [Iamp, i_el] = max(mINew, [], 1);
        tp = 0:1/CIParams.pps/maxima:(size(mINew,2)-1)/CIParams.pps/maxima;

        % --- AN model: o6 is the full 35000-fiber binary spike matrix ---
        [~,~,~,~,~,spikeMatrix] = main_CI_AN_model(Iamp, i_el, tp, CIParams.Tph, ...
            ANParams.N_nervecells, ANParams.xGroup, ANParams.fsZAS, ...
            ANParams.indexGroup, ANParams.hTP1, ANParams.Hd, ANParams);

        % --- extract spike times ---
        N_fibers = size(spikeMatrix, 1);
        spike_times = cell(N_fibers, 1);
        for i = 1:N_fibers
            spike_idx = find(spikeMatrix(i, :));
            spike_times{i} = spike_idx / fs_AN;  % seconds
        end

        % --- save ---
        if strcmp(side{1}, 'left')
            spike_times_left = spike_times;
            save(fullfile(savepath, 'spike_times_left.mat'), 'spike_times_left');
            fprintf('run_CI_cochlea: LEFT done — %d fibers.\n', N_fibers);
        else
            spike_times_right = spike_times;
            save(fullfile(savepath, 'spike_times_right.mat'), 'spike_times_right');
            fprintf('run_CI_cochlea: RIGHT done — %d fibers.\n', N_fibers);
        end
    end

    fprintf('run_CI_cochlea: complete.\n');
end