function [displaymatrix,display_t] = displayANpattern(logicalAN,tp)
% displays a stimulation pattern by summing up the stimulation activity in
% XSUM samples in order to match the sampling frequency of the matrix
% better with the resolution of the display

XSUM = 20;
frames = 1:floor(size(logicalAN,2)/XSUM);

for iCounter = 1:size(logicalAN,1)  %for each electrode
    for jCounter = frames
        displaymatrix(iCounter,jCounter) = sum(logicalAN(iCounter,(jCounter-1)*XSUM+1:jCounter*XSUM));
    end
end

display_t = [0:tp(2)*XSUM:tp(end)];
displaymatrix = sign(displaymatrix);
figure, imagesc(display_t,[0:5:35],displaymatrix);
axis xy
c = colormap('gray');
c = c(end:-1:1,:);
set(gcf,'Colormap',c);
xlabel('Time (s)');
ylabel('Cochlear location (mm)');

