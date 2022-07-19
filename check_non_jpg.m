%% Todo
% There are some image files are actually NOT jpeg files
% which will raise error warning in `vl_imreadjpeg`.
% Find out those fake jpeg files,
% and re-save them with python cv2/PIL.Image.
%% References
% - https://ch.mathworks.com/matlabcentral/answers/364719-detect-warning-and-take-action
function check_non_jpg(varargin)
    args = inputParser;
    addOptional(args, 'image_path', "/home/dataset/nuswide/Flickr");
    addOptional(args, 'matconvnet_path', "/usr/local/matconvnet-1.0-beta25");
    addOptional(args, 'o', "fake-jpg.txt");
    parse(args, varargin{:});

    this_path = cd(args.Results.matconvnet_path);
    setup;
    cd(this_path);
    warning(''); % clear old warning message

    log_file = fopen(args.Results.o, "w");
    [n_fake, n_file] = dfs(args.Results.image_path, log_file, 0);
    if 0 == n_fake
        fprintf(log_file, "NO FAKE JPEG FOUND\n");
    end
    fclose(log_file);
    % if 0 == n_fake
    %     fprintf("NO fake jpeg, deleting the output file: %s\n", args.Results.o);
    %     system(sprintf("rm %s", args.Results.o));
    % end
    exit
end % check_non_jpg


function [n_fake, file_cnt] = dfs(image_path, log_file, file_cnt)
    n_fake = 0;
    file_list = dir(image_path);
    for i = 1 : length(file_list)
        file_name = file_list(i).name;
        file_name_f = fullfile(image_path, file_name);
        if isfolder(file_name_f)
            if strcmp(file_name, ".") == 0 && strcmp(file_name, "..") == 0
                % fprintf("folder: %s\n", file_name_f);
                [fake_cnt, file_cnt] = dfs(file_name_f, log_file, file_cnt);
                n_fake = n_fake + fake_cnt;
            end
        elseif contains(file_name, ".jpg") || contains(file_name, ".jpeg")
            % fprintf("image file: %s\n", file_name_f);

            % tentative loading
            img = vl_imreadjpeg({char(file_name_f)});
            % if that's a fake jpeg file, a warning will be raised & catched below
            [warnMsg, warnId] = lastwarn;
            if ~isempty(warnMsg)
                fprintf("%s\n", file_name_f);
                fprintf(log_file, "%s\n", file_name_f);
                warning(''); % clear old warning message
                n_fake = n_fake + 1;
            end

            file_cnt = file_cnt + 1;
            if mod(file_cnt, 1000) == 0
                fprintf("%d\n", file_cnt);
            end
        end % if dir
    end % for file_list
end % dfs
