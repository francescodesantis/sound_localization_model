function hTP1 = generateImpRespPostMasking(fs)

t = -0.01:1/fs:0.01;

tauTP1 = 1e-3; % time constant of the impulse response (page 145)
% hTP1   = 1/(sqrt(2*pi)*tauTP1) * exp(-(t.^2)./(2*tauTP1^2));
hTP1   =                       1 * exp(-(t.^2)./(2*tauTP1^2));
