UN=zeros(10000,7216);
tic;

for i=1:10000
  UN(i,:)=membraneNoise(7216,7200);
end;

toc
save noise_CSR900_N8_1sec.mat UN