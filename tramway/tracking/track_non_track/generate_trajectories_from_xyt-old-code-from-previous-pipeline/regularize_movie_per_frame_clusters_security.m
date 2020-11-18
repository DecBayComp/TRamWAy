function movie_per_frame = regularize_movie_per_frame_clusters_security(movie_per_frame, dt_theo)


    t_tot            = [t_min:dt_theo:t_max];
    movie_per_frame2 = movie_per_frame;
    
    for i = 1 : length(t_tot)
       
        movie_per_frame2(i).x      = [];
        movie_per_frame2(i).y      = [];
        movie_per_frame2(i).t      = t_tot(i);
        movie_per_frame2(i).nb     = 0;
        
    end

    for i = 1 : length(movie_per_frame)
        
        index                      = round( ( movie_per_frame(i).t - t_min )/ dt_theo ) + 1;
        movie_per_frame2(index).x  = movie_per_frame(i).x;
        movie_per_frame2(index).y  = movie_per_frame(i).y;
        movie_per_frame2(index).nb = length(movie_per_frame2(index).y);
    end
        
    movie_per_frame = movie_per_frame2;

    
end