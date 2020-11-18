function movie_per_frame = convert_traj_struct(film)
warning off;

%% check if multiple processors can be used
% kill_parallel_multi_version;
% start_check_parallel_multi_version;
%%


[~,m]           = size(film);
movie_per_frame = struct;

if (m==3)


    x = film(:,1);
    y = film(:,2);
    t = film(:,3);

    T = unique(t);


    for i =1 : length(T)
    
        II  = find(  t == T(i) );
        movie_per_frame(i).x  = x(II);
        movie_per_frame(i).y  = y(II);
        movie_per_frame(i).t  = T(i);
        movie_per_frame(i).nb = length(II);
    
    end;

elseif (m==4)
    
    x = film(:,2);
    y = film(:,3);
    t = film(:,4);

    T = unique(t);


    for i =1 : length(T)
    
        II  = find(  t == T(i) );
        movie_per_frame(i).x  = x(II);
        movie_per_frame(i).y  = y(II);
        movie_per_frame(i).t  = T(i);
        movie_per_frame(i).nb = length(II);
    
    end;

else
    
    fprintf('check the input file ...\n');
    
    
end

%parametres = generate_parameters(movie_per_frame);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%







