function [assignment, tout,dt,parameters ] = Assignment_Multi_Mode_Full_Movie(movie_per_frame,modal, Previous_Assignment, name_pipeline, D_high, varargin)
%% give the comple assigment of a movie from movie per frame
%to be found in : Matlab/projet/Mapping_without_tracking/Assignment


if ~exist('Previous_Assignment');        Previous_Assignment        = []; end
if ~exist('name_pipeline');              name_pipeline              = 'main'; end

if isempty(Previous_Assignment)
    generate_global_warning_message_stop();

    %% check if multiple processors can be used
%     selective_kill_parallel_multi_version;
%     start_check_parallel_multi_version;

    parameters  = generate_parameters(movie_per_frame, [], name_pipeline, D_high);
    dt          = parameters.dt_theo; 
    n_movie     = parameters.n_movie_per_frame;
    tout        = [];


    for i =1 : n_movie -1;

        %fprintf('image %i\n', i);
        assignment(i)   = Assignment_Multi_Mode(movie_per_frame,parameters,i,modal);

    end;

    if parameters.d == 2

        for i =1 : n_movie -1;
            tout = [tout; assignment(i).x, assignment(i).y, assignment(i).t, assignment(i).dx, assignment(i).dy];
        end

    elseif parameters.d == 3

        for i =1 : n_movie -1;
            tout = [tout; assignment(i).x, assignment(i).y, assignment(i).z, assignment(i).t, assignment(i).dx, assignment(i).dy, assignment(i).dz];
        end

    end

else 
    
    parameters  = generate_parameters(movie_per_frame, [], name_pipeline, D_high);
    dt          = parameters.dt_theo; 
    n_movie     = parameters.n_movie_per_frame;
    tout        = [];
    assignment  = Previous_Assignment;
    
    
    if parameters.d == 2

        for i =1 : n_movie -1;
            tout = [tout; assignment(i).x, assignment(i).y, assignment(i).t, assignment(i).dx, assignment(i).dy];
        end

    elseif parameters.d == 3

        for i =1 : n_movie -1;
            tout = [tout; assignment(i).x, assignment(i).y, assignment(i).z, assignment(i).t, assignment(i).dx, assignment(i).dy, assignment(i).dz];
        end

    end
end