function trajs = filter_immobile_particles(trajs, sigma_eff)
%% quick filter


trajs_out   = [];
% sigma_eff   = 0.04;
dt          = trajs(2:end,4) - trajs(1:end-1,4);
II          = dt>0;
dt          = dt(II);
dt          = min(dt);

D_noise_eff = sigma_eff^2./dt;


nb_unique        = unique(trajs(:,1));
indice           = 1;
nb_unique_length =  length(nb_unique);
for j = 1 : nb_unique_length
    fprintf( '%i\t %i\n', j,nb_unique_length );
    %fprintf('%i\t %i\n', j, length(nb_unique));
    II    = trajs(:,1) == nb_unique(j);
    x     = trajs(II,2);
    y     = trajs(II,3);
    t     = trajs(II,4);
    dr2   = (x(2:end) - x(1:end-1)).^2 + (y(2:end) - y(1:end-1)).^2;
    dt    = t(2:end) - t(1:end-1);
    D_eff = mean(dr2./dt); 
    
    if (D_eff >= 1 * D_noise_eff)
       n_x = length(x); 
       trajs_out = [trajs_out; indice*ones(n_x,1), x,y,t]; 
       indice = indice + 1;
    end
    clear II x y t dr2 dt D_eff;
end



 trajs = trajs_out;







