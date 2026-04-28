% figureS3reproduce

N_nervecells = 1000; %Miller used 62 nerve fibers. However, for better averaging, I use 1000;
% Neural latency
[ML50,SIGMAL50] = generateNeuralLatencyConstants(N_nervecells);

P_AP = [0:.01:1];
for iCounter = 1:N_nervecells  %loop over each nerve cell
    [latency(iCounter,:),tmp,jitter(iCounter,:)] = calculate_latency_dj(ML50(iCounter),SIGMAL50(iCounter),P_AP);
end

figure, plot(P_AP,mean(latency,1).*1e6,'k-'); % plot the latency averaged over nervecells
ylim([0 1000]);
xlabel('spiking probability');
ylabel('latency, jitter (\mus)')

hold on;
% include latency data of Miller et al. (1999)
Miller_latency_x = [.0481 .101 .15 .202 .3 .401 .501 .6 .702 .801 .85 .899 .953 .992];
Miller_latency_y = [746   718   701 681 655 633  618 606 587 561  546  524  502  482];
latencydatahandle = plot(Miller_latency_x,Miller_latency_y,'ko');

modelhandle = plot(P_AP,mean(jitter,1).*1e6,'k-'); % plot the jitter averaged over nervecells
% include jitter data of Miller et al. (1999)
Miller_jitter_x = [.0527 .102 .152 .202 .301 .401 .501 .599 .699 .8 .846 .897 .948 .987];
Miller_jitter_y = [ 134  131  128  124  122  118  115  107 98.6  87 78.3  64  51.1  43.9];
jitterdatahandle = plot(Miller_jitter_x,Miller_jitter_y,'ks');

legend([latencydatahandle, jitterdatahandle, modelhandle], 'latency (corrected for Tph)','jitter, both from Miller et al.(1999)','model functions');