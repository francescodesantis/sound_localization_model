function [displaymatrix,display_t] = displaystimulationpattern(mINew,tp)
% displays a stimulation pattern by summing up the stimulation activity in
% XSUM samples in order to match the sampling frequency of the matrix
% better with the resolution of the display

XSUM = 20;
frames = 1:floor(size(mINew,2)/XSUM);

for iCounter = 1:size(mINew,1)  %for each electrode
    for jCounter = frames
        displaymatrix(iCounter,jCounter) = sum(mINew(iCounter,(jCounter-1)*XSUM+1:jCounter*XSUM));
    end
end

display_t = [0:tp(2)*XSUM:(size(displaymatrix,2)-1)*tp(2)*XSUM];

figure, imagesc(display_t,[],displaymatrix);
axis xy
c = colormap('gray');
c = c(end:-1:1,:);
set(gcf,'Colormap',c);
xlabel('Time (s)');
ylabel('Electrode no.');
