function I = conversion_CL2Iamp(cl,szWhichDevice)




switch szWhichDevice
    case 'nucleus'
        I = 10.175.^(cl/255); % aus laneau
        % I = I/24.838;
    case 'CI22'
        I = 100e-6 * 1.0205.^(cl-100); % aus hearcom website
    case 'CI24M'
        I = 100e-6 * 1.0205.^(cl-100); % aus hearcom website
    case 'CI24R'
        I = 100e-6 * 1.0205.^(cl-100); % aus hearcom website
    case 'freedom' % also CI24RE
        I = 100e-6 * 1.0182.^(cl-100); % aus hearcom website
    otherwise
        error('this device is not known')
end