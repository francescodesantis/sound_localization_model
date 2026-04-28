function y = process_preemphasis_CI(x)


global FS;

% FS = 16000;
% x = zeros(4096,1);
% x(1) = 1;

fc = 1200; 

w = 2*fc/FS;


[b,a] = butter(1,w,'high');


y = filter(b,a,x);

df = FS/length(x);
vF = 0:length(x)-1;
vF = vF*df;
 
% figure,semilogx(vF,20*log10(abs(fft(y))))
% figure,cohere(x,y)