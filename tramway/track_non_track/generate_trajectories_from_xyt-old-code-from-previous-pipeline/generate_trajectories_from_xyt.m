function trajs = generate_trajectories_from_xyt(name, D_high, sigma_eff, varargin)

    n_cut              = 25000;
    name_pipeline      = 'global_maps';
    xyt                = load(name);
    xyt_pieces         = split_trajs_in_pieces(xyt, n_cut);
    n_pieces           = length(xyt_pieces);
    trajs              = cell(n_pieces, 1);
    trajs_clean        = cell(n_pieces, 1);
    
    n_init             = 0;
    n_init_clean       = 0;
%     for i = 1 : 3
    for i = 1 : n_pieces
        fprintf('%i\t %i\n', i,n_pieces);
        movie_per_frame    = convert_traj_struct(xyt_pieces{i,1}); 
%         fprintf('here 1\n');
        [assignment, ~,~ ] = Assignment_Multi_Mode_Full_Movie(movie_per_frame,'hungarian',[], name_pipeline, D_high);
%          fprintf('here 2\n'); 
        if i>1
            n_init = trajs_loc(end,1);
            n_init_clean = trajs_loc_clean(end,1);
        end
        [trajs_loc]        = traj_from_assignment(assignment, 2,n_init); 
        [trajs_loc_clean]  = traj_from_assignment_filter_immobiles(assignment, 2,n_init_clean, sigma_eff);
%           fprintf('here 3\n');
        trajs{i,1}         = trajs_loc; 
        trajs_clean{i,1}   = trajs_loc_clean; 
        
    end
    
    n_trajs         = length(trajs);
    n_trajs_clean   = length(trajs_clean);
    trajs_tot       = [];
    trajs_tot_clean = [];
    
    
    for i= 1 : n_trajs
       trajs_tot       = [trajs_tot; trajs{i,1}];
    end
    
    for i= 1 : n_trajs_clean
       trajs_tot_clean = [trajs_tot_clean; trajs_clean{i,1}];
    end
    
    
    clear trajs trajs_loc trajs_clean;
    
    
    trajs       = trajs_tot;
    trajs_clean =  trajs_tot_clean;
    clear trajs_tot  trajs_tot_clean;
    dlmwrite(['trajectories_' name],trajs,'-append','roffset',0,'delimiter',' ');
    dlmwrite(['trajectories_clean_' name],trajs_clean,'-append','roffset',0,'delimiter',' ');
    
    
    
    
    
    
%     n_trajs = length(trajs(:,1));
%     fichier = fopen(['trajectories_' name], 'w+');
%     for i = 1 : n_trajs
%        fprintf(fichier, '%i\t %f\t %f\t %f\n', trajs(i,1),trajs(i,2),trajs(i,3),trajs(i,4) ); 
%     end
%     fclose(fichier);
%%  raw files

    
%%  clean trajectories    
%     trajs_clean   = filter_immobile_particles(trajs, sigma_eff);
%     n_trajs_clean = length(trajs_clean(:,1));
%     fichier = fopen(['trajectories_clean_' name], 'w+');
%     for i = 1 : n_trajs_clean
%        fprintf(fichier, '%i\t %f\t %f\t %f\n', trajs_clean(i,1),trajs_clean(i,2),trajs_clean(i,3),trajs_clean(i,4) ); 
%     end
%     fclose(fichier);    
%     
    
    
end
%     movie_per_frame = regularize_movie_per_frame_clusters_security(movie_per_frame, dt_theo);
%     clusters(i).movie_per_frame = movie_per_frame ;