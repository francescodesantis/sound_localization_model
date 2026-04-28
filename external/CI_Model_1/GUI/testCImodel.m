function varargout = testCImodel(varargin)
% TESTCIMODEL M-file for testCImodel.fig
%      TESTCIMODEL, by itself, creates a new TESTCIMODEL or raises the existing
%      singleton*.
%
%      H = TESTCIMODEL returns the handle to a new TESTCIMODEL or the handle to
%      the existing singleton*.
%
%      TESTCIMODEL('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in TESTCIMODEL.M with the given input arguments.
%
%      TESTCIMODEL('Property','Value',...) creates a new TESTCIMODEL or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before testCImodel_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to testCImodel_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help testCImodel

% Last Modified by GUIDE v2.5 27-Nov-2013 15:28:00

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @testCImodel_OpeningFcn, ...
                   'gui_OutputFcn',  @testCImodel_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before testCImodel is made visible.
function testCImodel_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to testCImodel (see VARARGIN)
addpath(['..' filesep 'parameterstore']);
addpath(['..']);
% Choose default command line output for testCImodel
handles.output = hObject

% Update handles structure
guidata(hObject, handles);
% set(handles.pushbutton_ECAP_Growth,'enable','off');
% set(handles.pushbutton_ECAP_Recovery,'enable','off');
% set(handles.pushbuttonECAP_SEQ_Const,'enable','off');
% set(handles.pushbutton_ECAP_SEQ_mod,'enable','off');
% set(handles.pushbutton_single_ECAP,'enable','off');
% set(handles.Template_generation,'enable','off');
% set(handles.pushbutton_classification,'enable','off');
% set(handles.Electric_signals,'enable','off');
% set(handles.pushbutton1,'String','Spatial Spread');
% set(handles.pushbutton2,'String','Compression Characteristic');

% UIWAIT makes testCImodel wait for user response (see UIRESUME)
% uiwait(handles.figure1);
[image, map] = imread('H4A_Logo_Web_Office.jpg', 'jpeg');
imshow(image,map);

% --- Outputs from this function are returned to the command line.
function varargout = testCImodel_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% PLOT SPATIAL SPREAD
CIParams.devicename = get(handles.edit1,'String');
set_global_constants
figure;
for iCounter = 1:size(ANParams.V,2)
    plot(ANParams.X_NZ,ANParams.V(:,iCounter));
    hold on;
    xlabel('length of cochlear partition (mm)')
    ylabel('voltage (arbitrary units)')
    title('Spatial spread of the electrical field alongside the electrode array')
end
hold on
vertical_location_electrode = max(max(ANParams.V))*1.1;
plot([0 ANParams.X_EL], repmat(vertical_location_electrode,size(ANParams.X_EL,1),size(ANParams.X_EL,2)+1),'k-','LineWidth',8);
plot(ANParams.X_EL, repmat(vertical_location_electrode,size(ANParams.X_EL,1),size(ANParams.X_EL,2)),'ko-','MarkerFaceColor',[1 1 1],'MarkerSize',8);

% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% PLOT COMPRESSION CHARACTERISTIC
addpath(['..' filesep 'ACE']);
CIParams = set_global_constants_ACE_demo();
A = [0:0.0001:1]; %input current amplitude in mA(?)
C = process_compression_ci(A,CIParams.B,CIParams.M,CIParams.alpha_c);
figure;
plot(A,C);
xlabel('input current amplitude (arb. units)');
ylabel('output current amplitude (arb. units)');

function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double

% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on selection change in popupwavfile.
function popupwavfile_Callback(hObject, eventdata, handles)
% hObject    handle to popupwavfile (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Hints: contents = cellstr(get(hObject,'String')) returns popupwavfile contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupwavfile

% --- Executes during object creation, after setting all properties.
function popupwavfile_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupwavfile (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
 % Set wav files in wave-dir as Strings
 wavfiles = dir(fullfile('..', 'wavfilestore', '*.wav'));
 set(hObject, 'String', char(wavfiles.name));

% --- Executes on button press in runmodelbutton.
function runmodelbutton_Callback(hObject, eventdata, handles)
% hObject    handle to runmodelbutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% perform cleanup of parameterstructs
clear ANPArams CIParams;
% Get parameterfile
paramfilename = get(handles.popupparameterfile, 'String');
paramfilename = strtrim(paramfilename(get(handles.popupparameterfile, 'Value'),:)); %strtrim removes trailing whitespaces
paramfilename = ['..' filesep 'parameterstore' filesep paramfilename];
% get wav-filename for processing
signalfilename = get(handles.popupwavfile, 'String');
signalfilename = strtrim(signalfilename(get(handles.popupwavfile, 'Value'),:)); %strtrim removes trailing whitespaces
signalfilename = ['..' filesep 'wavfilestore' filesep signalfilename]; % constructs full path to filename
% Call the main modelling function
main_SP_CI_AN(signalfilename, paramfilename);

% --- Executes on selection change in popupparameterfile.
function popupparameterfile_Callback(hObject, eventdata, handles)
% hObject    handle to popupparameterfile (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupparameterfile contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupparameterfile

% --- Executes during object creation, after setting all properties.
function popupparameterfile_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupparameterfile (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
% Set parameter files in parameterstore-dir as Strings
 parameterfiles = dir(fullfile('..', 'parameterstore', '*.m'));
 set(hObject, 'String', char(parameterfiles.name));
 
 % Set basedir for speech prediction olsa-files
setenv('basedir', ['..' filesep '..' filesep 'CImodelresults' filesep 'results' filesep]);

% --- Executes on button press in buttonreproduceFig3.
function buttonreproduceFig3_Callback(hObject, eventdata, handles)
% hObject    handle to buttonreproduceFig3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
figure3reproduce;

% --- Executes on button press in buttonreproduceFig4.
function buttonreproduceFig4_Callback(hObject, eventdata, handles)
% hObject    handle to buttonreproduceFig4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
figure4reproduce;

% --- Executes on button press in buttonreproduceFigS3.
function buttonreproduceFigS3_Callback(hObject, eventdata, handles)
% hObject    handle to buttonreproduceFigS3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
figureS3reproduce;

% --- Executes on button press in ButtonreproduceFigS2.
function ButtonreproduceFigS2_Callback(hObject, eventdata, handles)
% hObject    handle to ButtonreproduceFigS2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
figureS2reproduce;

% --- Executes on button press in ButtonreproduceFigS4.
function ButtonreproduceFigS4_Callback(hObject, eventdata, handles)
% hObject    handle to ButtonreproduceFigS4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
figureS4reproduce;

% --- Executes during object creation, after setting all properties.
function figure1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
% add needed paths (parent directory with all subfolders)
addpath(genpath(['..' filesep]));

% --- Executes during object deletion, before destroying properties.
function figure1_DeleteFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% remove needed paths (parent directory with all subfolders)
rmpath(genpath(['..' filesep]));

% % --- Executes on button press in pushbutton_ECAP_Growth.
% function pushbutton_ECAP_Growth_Callback(hObject, eventdata, handles)
% % hObject    handle to pushbutton_ECAP_Growth (see GCBO)
% % eventdata  reserved - to be defined in a future version of MATLAB
% % handles    structure with handles and user data (see GUIDATA) 
% ECAP_growth;
% 
% % --- Executes on button press in pushbutton_ECAP_Recovery.
% function pushbutton_ECAP_Recovery_Callback(hObject, eventdata, handles)
% % hObject    handle to pushbutton_ECAP_Recovery (see GCBO)
% % eventdata  reserved - to be defined in a future version of MATLAB
% % handles    structure with handles and user data (see GUIDATA)
% ECAP_recovery;
% 
% % --- Executes on button press in pushbuttonECAP_SEQ_Const.
% function pushbuttonECAP_SEQ_Const_Callback(hObject, eventdata, handles)
% % hObject    handle to pushbuttonECAP_SEQ_Const (see GCBO)
% % eventdata  reserved - to be defined in a future version of MATLAB
% % handles    structure with handles and user data (see GUIDATA)
% ECAP_sequence_const;
% 
% % --- Executes on button press in pushbutton_ECAP_SEQ_mod.
% function pushbutton_ECAP_SEQ_mod_Callback(hObject, eventdata, handles)
% % hObject    handle to pushbutton_ECAP_SEQ_mod (see GCBO)
% % eventdata  reserved - to be defined in a future version of MATLAB
% % handles    structure with handles and user data (see GUIDATA)
% ECAP_sequence_mod;
% 
% 
% % --- Executes on button press in pushbutton_single_ECAP.
% function pushbutton_single_ECAP_Callback(hObject, eventdata, handles)
% % hObject    handle to pushbutton_single_ECAP (see GCBO)
% % eventdata  reserved - to be defined in a future version of MATLAB
% % handles    structure with handles and user data (see GUIDATA)
% single_ECAP;

% --- Executes during object creation, after setting all properties.
function buttonreproduceFig4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to buttonreproduceFig4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
% set custom tooltip with line breaks:
set(hObject, 'TooltipString', sprintf('Model simulation of forward masking. Each figure contains a plot of the internal response of masker and target.\n Additionally the model function to simulate the masking threshold are plotted.\n This reproduces Figure 4 from Fredelake et al.'));

% --- Executes during object creation, after setting all properties.
function buttonreproduceFig3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to buttonreproduceFig3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
set(hObject, 'TooltipString', sprintf('Plots the spike probability of a single auditory nerve fiber over\n different input levels in dB and different pulses rates(pps).\n This reproduces Figure 3 from Fredelake et al.'));

% --- Executes during object creation, after setting all properties.
function ButtonreproduceFigS2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ButtonreproduceFigS2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
set(hObject, 'TooltipString', sprintf('Plots the refractory function,\n which models the increase in threshold while the auditory nerve fiber is in its refractroy state.\n This reproduces Figure 2 from Fredelake et al, supplement.'));

% --- Executes during object creation, after setting all properties.
function ButtonreproduceFigS4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ButtonreproduceFigS4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
set(hObject, 'TooltipString', sprintf('Plots the Number of action potentials over\n different Inter-Stimulus-Intevalls (ISI) for different pulse rates.\n This reproduces Figure 4 from Fredelake et al, supplement.'));

% --- Executes during object creation, after setting all properties.
function buttonreproduceFigS3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to buttonreproduceFigS3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
set(hObject, 'TooltipString', sprintf('Plots latency and jitter as a function of Stimulus Level, transformed to spiking probability\n This reproduces Figure 3 from Fredelake et al, supplement.'));

% % --- Executes on button press in Template_generation.
% function Template_generation_Callback(hObject, eventdata, handles)
% % hObject    handle to Template_generation (see GCBO)
% % eventdata  reserved - to be defined in a future version of MATLAB
% % handles    structure with handles and user data (see GUIDATA)
% 
% % Run template generator
% main_CI_AN_model_isaar;
% 
% % --- Executes on button press in pushbutton_classification.
% function pushbutton_classification_Callback(hObject, eventdata, handles)
% % hObject    handle to pushbutton_classification (see GCBO)
% % eventdata  reserved - to be defined in a future version of MATLAB
% % handles    structure with handles and user data (see GUIDATA)
% 
% % Run dtw-classificator
% main_dtw_Olsa_isaar;
% % --- Executes on button press in Electric_signals.
% function Electric_signals_Callback(hObject, eventdata, handles)
% % hObject    handle to Electric_signals (see GCBO)
% % eventdata  reserved - to be defined in a future version of MATLAB
% % handles    structure with handles and user data (see GUIDATA)
% 
% % generate electric waveforms
% main_generate_electric_signals_isaar;

% --- Executes during object creation, after setting all properties.
function Logo_Axes_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Logo_Axes (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate Logo_Axes
axes(hObject);
[image, map] = imread('H4A_Logo_Web_Office.jpg', 'jpeg');
imshow(image,map);

% --- Executes during object creation, after setting all properties.
function Electric_signals_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Electric_signals (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
set(hObject, 'TooltipString', sprintf('Generates electric signals from speech sentence database\n Needed for the second button to work.'));

% --- Executes during object creation, after setting all properties.
function Template_generation_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Template_generation (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
set(hObject, 'TooltipString', sprintf('Creates internal representation templates for speech predicition\n Needed for the third button to work.'));

% --- Executes during object creation, after setting all properties.
function pushbutton_classification_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pushbutton_classification (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
set(hObject, 'TooltipString', sprintf('Performs speech predicition using a DTW-Classifier\n and the templates generated in the second step.'));

% --- Executes during object creation, after setting all properties.
function pushbutton1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
set(hObject, 'TooltipString', sprintf('Calculates the modelled spatial spread function of the electric current for the given CI-Device.\n Possible choices are:\n ''nucleus''\n ''CI22''\n ''CI24M''\n ''CI24R''\n ''freedom'''));

% --- Executes during object creation, after setting all properties.
function pushbutton2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
set(hObject, 'TooltipString', sprintf('Displays the compression characteristic of the choosen CI-Device.\n '));
