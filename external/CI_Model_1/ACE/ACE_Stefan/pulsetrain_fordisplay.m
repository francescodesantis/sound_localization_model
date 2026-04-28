function train = pulsetrain_fordisplay(ppspattern,pps,maxima,sfreq_pulsepattern)

global ipg
global Tph


number_of_total_samples = round(sfreq_pulsepattern/pps/maxima*(size(ppspattern,2)+1));
train=zeros(1,number_of_total_samples);

%this is one biphasic pulse, which is used as a template
biphasic_pulse = [ones(1,round(Tph*sfreq_pulsepattern)) ... 
    zeros(1,round(ipg*sfreq_pulsepattern)) ones(1,round(Tph*sfreq_pulsepattern)).*(-1)];


for iCounter = 1:size(ppspattern,2)
    if ppspattern(iCounter) > 0
        startsample = round(sfreq_pulsepattern/pps/maxima*iCounter);
        train(startsample:startsample+length(biphasic_pulse)-1) = biphasic_pulse.*ppspattern(iCounter);
    end
end


