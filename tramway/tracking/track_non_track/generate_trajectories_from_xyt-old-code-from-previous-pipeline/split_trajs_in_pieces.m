function trajs_pieces = split_trajs_in_pieces(trajs, n_cut)



n_overlapp = 50; 
n_traj     = length(trajs(:,1));
n_pieces   = floor(n_traj./(n_cut- n_overlapp)); 
trajs_pieces = cell(n_pieces,1);

for i = 1 : n_pieces
    init = 1 +(i-1)* (n_cut - n_overlapp);
    fin  =  (i-1)* (n_cut - n_overlapp) + n_cut; 
    trajs_loc = trajs(init:fin, :);
    trajs_pieces{i,1} = trajs_loc;
    clear trajs_loc;
    
end










end

