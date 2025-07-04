%% Scanbox file loading script
% Select a .sbx file using uigetfile, load the associated .mat file with
% the info struct

% select the .sbx file
[sbxName,sbxLocation] = uigetfile("*.sbx","Select Scanbox File");
% load the "info" structure
baseName = strsplit(sbxName,'.sbx');
baseName = baseName{1}; % save baseName for the videowriter later
infoName = [baseName, '.mat'];
load(fullfile(sbxLocation,infoName));
    
if(info.scanmode==0)
    info.recordsPerBuffer = info.recordsPerBuffer*2;
end

% for the "factor" variable.
factor = 3-info.chan.nchan; % 2 if nchans is 1, and vice versa


% get a file pointer and information about the file
info.fid = fopen(fullfile(sbxLocation,sbxName));
d = dir(fullfile(sbxLocation,sbxName));
info.bpr = (info.sz(2) * info.recordsPerBuffer * 2 * info.chan.nchan);   % bytes per record 

% max index value and bytes per record
info.bpr = (info.sz(2) * info.recordsPerBuffer * 2 * info.chan.nchan);   % bytes per record -- uint16 so two bytes
info.maxIdx = d.bytes/info.bpr; % highest index allowed

% read the file
x = fread(info.fid,'uint16=>uint16'); % read it
x = reshape(x,[info.chan.nchan info.sz(2) info.recordsPerBuffer  info.maxIdx]); % reshape
x = intmax('uint16')-permute(x,[1 3 2 4]); % swap the vertical and horizontal, flip the sign (sort of)
% x = double(x)/double(quantile(x(:),.9));

% save to video
f = figure;
v = VideoWriter(string(fullfile(sbxLocation, [baseName, '.avi'])));
open(v);
for jj = 1:info.maxIdx
    for ii = 1:info.chan.nchan
        subplot(info.chan.nchan,1, ii)
        imagesc(squeeze(x(ii, :, :, jj)))
        % image(squeeze(x(ii, :, :, jj)))
        box off
        axis off
        colormap('gray')
        writeVideo(v,getframe(f))
    end
end
% always close your open handles ;)
close(v)
close(f)


