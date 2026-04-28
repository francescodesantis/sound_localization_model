%% check ACE-Constants for sensible input
% MHL and THL should have the same length
if size(CIParams.MCL) ~= size(CIParams.TCL)
    errordlg('There must be the same number of M-Levels (CIParams.MCL) and T-Levels (CIParams.TCL). Please check the Parameterfile!');
    return;
end
% T levels should be smaller than M Levels
if CIParams.TCL >= CIParams.MCL
    errordlg('T-Levels (CIParams.TCL) must be smaller than M-Levels (CIParams.MCL). Please check the Parameterfile!');
    return;
end
% CIParams.maxima should be smaller than CIParams.NumOfChannels
if CIParams.maxima > CIParams.NumOfChannels
    errordlg('In a n- of m-Strategie n (CIParams.maxima) must be smaller than m (CIParams.NumOfChannels). Please check the Parameterfile!');
    return;
end
% CIParams.T_SPL should be smaller than CIParams.C_SPL
if CIParams.T_SPL >= CIParams.C_SPL
    errordlg('T-Levels (CIParams.T_SPL) must be smaller than C-Levels (CIParams.C_SPL). Please check the Parameterfile!');
    return;
end
% Currently only exactly 22 Electrodes are supported
if CIParams.NumOfChannels ~= 22
    errordlg('Electrode Number (CIParams.NumOfChannels) must be 22. Other Electrode numbers are noch supported at this time. Please check the Parameterfile!');
    return;
end