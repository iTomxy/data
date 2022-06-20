%% Todo
% There are some image files are actually NOT jpeg files
% which will raise error warning in `vl_imreadjpeg`.
% Find out those fake jpeg files,
% and re-save them with python cv2/PIL.Image.
%% References
% - https://ch.mathworks.com/matlabcentral/answers/364719-detect-warning-and-take-action
function check_non_jpg(varargin)
    args = inputParser;
    addOptional(args, 'image_path', "/home/dataset/nuswide/images");
    addOptional(args, 'matconvnet_path', "/home/tom/Download/matconvnet-1.0-beta25");
    addOptional(args, 'o', "fake-jpg.txt");
    parse(args, varargin{:});

    this_path = cd(args.Results.matconvnet_path);
    setup;
    cd(this_path);
    warning(''); % clear old warning message

    file_list = dir(fullfile(args.Results.image_path, "*.jpg"));
    fprintf("#images: %d\n", length(file_list));
    n_fake = 0;  % count fake jpeg files
    log_file = fopen(args.Results.o, "w");
    for i = 1 : length(file_list)
        img_file = file_list(i).name;
        % sid = split(img_file, ".jpg");
        % sid = str2num(sid{1});

        % tentative loading
        img = vl_imreadjpeg({char(fullfile(args.Results.image_path, img_file))});
        % if that's a fake jpeg file, a warning will be raised & catched below
        [warnMsg, warnId] = lastwarn;
        if ~isempty(warnMsg)
            % format: <image ID>.jpg
            fprintf("%s\n", img_file);
            fprintf(log_file, "%s\n", img_file);
            warning(''); % clear old warning message
            n_fake = n_fake + 1;
        end

        if mod(i, 1000) == 0
            fprintf("%d\n", i);
        end
        % break
    end
    if 0 == n_fake
        fprintf(log_file, "NO FAKE JPEG FOUND\n");
    end
    fclose(log_file);
    % if 0 == n_fake
    %     fprintf("NO fake jpeg, deleting the output file: %s\n", args.Results.o);
    %     system(sprintf("rm %s", args.Results.o));
    % end
end % test_non_jpg
