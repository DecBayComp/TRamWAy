function [out ] = Assignment_Multi_Mode(test,parameters,i,mode,varargin)


format long;

if ( (nargin<=2 )  )
   out = [];
   return;
end
   
switch mode
    
    case 'bipartite'
        %% solve using min sum
        try
            out = min_sum_solution(test,parameters,i);
        catch
            out = junk_solution(test,parameters,i);
        end
    case 'hungarian'  
        %% solve using kuhn munktree algortihm for non bipartite graph
        try
            out = hungarian(test,parameters,i);
        catch
            out = junk_solution(test,parameters,i);
        end
    case 'Min_Sum_1'
        %% solve using min sum approx 1 
        try
            out = approx_min_sum_solution_1(test,parameters,i);
        catch
            out = junk_solution(test,parameters,i);
        end
    case  'Min_Sum_2'  
        %% solve using min sum approx 2
        try
            out = approx_min_sum_solution_2(test,parameters,i);
        catch
            out = junk_solution(test,parameters,i);
        end
end

    
end
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                       Bipartite solution Min-Sum                      %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [out] = min_sum_solution(test,parameters,i)

            
[log_likelihood_matrix, nn]                             = compute_all(test, i, parameters);
[boolean_matrix1, boolean_matrix2]                      = generate_boolean(nn) ;
II                                                      = 1:nn;
[x_A, x_B]                                              = updates_as_bipartite( nn, log_likelihood_matrix, parameters,boolean_matrix1, boolean_matrix2, II);
[~,~, binary_localization, real_assignment, ~, state ]  = find_optimal_assigment(x_A,x_B, nn, log_likelihood_matrix , II);
cost_function_Value                                     = cost_function(log_likelihood_matrix, binary_localization);  
        
t                            = test(i).t;    
x1                           = test(i).x;
y1                           = test(i).y; 
x2                           = test(i+1).x;
y2                           = test(i+1).y;
out.t                        = t;
% out.x1                    = x1;
% out.y1                    = y1;
% out.x2                    = x2;
% out.y2                    = y2;
if (parameters.d == 3)
    z1                       = test(i).z;
    z2                       = test(i+1).z;
%     out.z1                   = z1;
%     out.z2                   = z2;
end

% out.x = x1;
% out.y = y1;
% if (parameters.d == 3)
%     out.z                 = z1;
% end

JJ                           = real_assignment ~= 0;   


out.x_all = x1;
out.y_all = y1;
out.x     = x1(JJ);
out.y     = y1(JJ);

if (parameters.d == 3)
    out.z                 = z1(JJ);
    out.z_all             = z1;
end
out.t                     = t.*ones(length(out.x),1);
out.t_all                 = t.*ones(length(out.x_all),1);


out.dx                    = x2(real_assignment(JJ)) - x1(JJ);
out.dy                    = y2(real_assignment(JJ)) - y1(JJ);
if (parameters.d == 3)
    out.dz                = z2(real_assignment(JJ)) - z1(JJ);
end

if isempty(II(JJ)')
%    fprintf('Here NaN \n');
    out.index_assigned           = NaN;
    out.real_assignment          = NaN;
    out.assigment                = NaN;
else
 %   fprintf('Here Normal \n');
    out.index_assigned           = II(JJ)';
    out.real_assignment          = real_assignment;
    out.assigment                = binary_localization;
end

% out.state                    = state;
% out.cost                     = cost_function_Value;
% out.log_lik                  = log_likelihood_matrix;

end
%%
function [log_likelihood_matrix, nn] = compute_all(test,kk, parameters)

nn = length(test(kk).x(:));

if (parameters.d ==2)
    
    dx                      = bsxfun(@minus, test(kk).x(:), test(kk+1).x(:)');
    dy                      = bsxfun(@minus, test(kk).y(:), test(kk+1).y(:)');
    distance_matrix         = dx.^2 + dy.^2;    

else 
    
    dx                      = bsxfun(@minus, test(kk).x(:), test(kk+1).x(:)');
    dy                      = bsxfun(@minus, test(kk).y(:), test(kk+1).y(:)');
    dz                      = bsxfun(@minus, test(kk).z(:), test(kk+1).z(:)');
    distance_matrix         = dx.^2 + dy.^2 + dz.^2;
    
end    
    
    log_likelihood_matrix = -distance_matrix;

end
%%
function [boolean_matrix1, boolean_matrix2] = generate_boolean(nn) 



boolean_matrix1  = ones(nn, nn, nn);
boolean_matrix2  = ones(nn, nn, nn);
for i =1 : nn;
    boolean_matrix1(i,:,i) = NaN;
    boolean_matrix2(i,:,i) = NaN;
end;


end
%%
function [x_A, x_B] =  initialisation(kind_init, nn)

switch lower(kind_init)
    
    case 'zeros'
        [x_A, x_B] = pure_0_initialize(nn);
    case 'random'
        [x_A, x_B] = random_0_1_initialize(nn);
    case 'normal'
        [x_A, x_B] = normal_initialize(nn); 
             
end;

end
%%
function [x_A, x_B] = random_0_1_initialize(nn)

    x_A = rand(nn);
    x_B = rand(nn);

end
%%
function [x_A, x_B] = pure_0_initialize(nn)

    x_A = zeros(nn);
    x_B = zeros(nn);

end
%%
function [x_A, x_B] = normal_initialize(nn)

    x_A = randn(nn);
    x_B = randn(nn);

end
%%
function [x_A, x_B] = updates_as_bipartite( nn, log_likelihood_matrix, parameters,boolean_matrix1, boolean_matrix2, II)

number_loop         = parameters.number_stop ;                       
nb_duration_state_1 = parameters.nb_duration_state_1;
global_state        = zeros(nb_duration_state_1,1);
kind_init           = parameters.kind_init  ;
index               = 0;
[x_A_old, x_B_old]  = initialisation(kind_init,nn);

   while ( floor( sum( global_state )./nb_duration_state_1)~=1  && (index < number_loop) )

         x_A_new  = reshape(  min( bsxfun(@times, boolean_matrix1, -log_likelihood_matrix  - x_B_old  ),[],1),nn,nn ) ;
         x_B_new  = reshape(  min( bsxfun(@times, boolean_matrix2, -log_likelihood_matrix' - x_A_old  ),[],1),nn,nn ) ;        

        [~,~, ~, real_assignment, number_assigment, state ] = find_optimal_assigment(x_A_new,x_B_new, nn, log_likelihood_matrix , II);        
        global_state(mod(index,nb_duration_state_1)+1,1)   = state;
    
        x_A_old  = x_A_new;
        x_B_old  = x_B_new;
    
        index   = index +1;
       % %fprintf('%i\t %i\t %i\n ', index, number_loop, state);
    end;
    
 x_A = x_A_new;
 x_B = x_B_new;

%fprintf('out bipartite %i\n ', index); 
 
end
%%
function [nij,nji, binary_localization, real_assignment, number_assigment, state ] = find_optimal_assigment(x_A,x_B, nn, log_likelihood_matrix , II)

[~, nij]                                           = min( -log_likelihood_matrix  - x_B , [] , 2 );
[~, nji]                                           = min( -log_likelihood_matrix' - x_A , [] , 2 );

binary_localization                                = false(nn);
status                                             = II' ==  nji(nij(II));

binary_localization( ( nij(II) - 1 ).*nn + II' )   = status ;
real_assignment                                     = nij(II).*( status ) ;
number_assigment                                   = sum (status);

state                                              = number_assigment == nn;

end
%%
function res = cost_function(log_likelihood_matrix, binary_localization)

CC = log_likelihood_matrix.*binary_localization;
res = sum( CC(:) )  ;

end
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                       End solution Min-Sum                            %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                       Hungarian Solution                              %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out =  hungarian(test,parameters,i)

[distance_matrix, distance_matrix_reduced, nn, row_reduced,col_reduced] = compute_all_hungarian(test, i, parameters);
[n,m]         = size(distance_matrix);
[binary_localization, real_assignment, index_global]              = kuhn_munktree(distance_matrix_reduced,nn,row_reduced,col_reduced, n, m);
[cost]       = cost_function_distance(distance_matrix, binary_localization ) ;


II                           = 1:length(distance_matrix(:,1));  

t                            = test(i).t; 
x1                           = test(i).x;
y1                           = test(i).y; 
x2                           = test(i+1).x;
y2                           = test(i+1).y;
% out.x1                    = x1;
% out.y1                    = y1;
% out.x2                    = x2;
% out.y2                    = y2;
if (parameters.d == 3)
    z1                       = test(i).z;
    z2                       = test(i+1).z;
%     out.z1                   = z1;
%     out.z2                   = z2;
end


JJ                              = real_assignment ~= 0;  
%II(JJ)
%index_assigned                 = II(JJ)';



out.x_all = x1;
out.y_all = y1;
out.x     = x1(JJ);
out.y     = y1(JJ);

if (parameters.d == 3)
    out.z                 = z1(JJ);
    out.z_all             = z1;
end
out.t                        = t.*ones(length(out.x),1);
out.t_all                 = t.*ones(length(out.x_all),1);

out.dx                    = x2(real_assignment(JJ)) - x1(JJ);
out.dy                    = y2(real_assignment(JJ)) - y1(JJ);
if (parameters.d == 3)
    out.dz                = z2(real_assignment(JJ)) - z1(JJ);
end

if isempty(II(JJ)')
%    fprintf('Here NaN \n');
    out.index_assigned           = NaN;
    out.real_assignment          = NaN;
    out.assigment                = NaN;
else
 %   fprintf('Here Normal \n');
    out.index_assigned           = II(JJ)';
    out.real_assignment          = real_assignment;
    out.assigment                = binary_localization;
end

% out.cost                     = cost;
% out.dist_mat                 = distance_matrix;
% out.nb_iteration             = index_global;

end
%%
function [distance_matrix, distance_matrix_reduced, nn, row_reduced,col_reduced] = compute_all_hungarian(test, kk, parameters)
    
    nn = length(test(kk).x(:));

if (parameters.d ==2)
    
    dx                      = bsxfun(@minus, test(kk).x(:), test(kk+1).x(:)');
    dy                      = bsxfun(@minus, test(kk).y(:), test(kk+1).y(:)');
    distance_matrix         = dx.^2 + dy.^2;    

else 
    
    dx                      = bsxfun(@minus, test(kk).x(:), test(kk+1).x(:)');
    dy                      = bsxfun(@minus, test(kk).y(:), test(kk+1).y(:)');
    dz                      = bsxfun(@minus, test(kk).z(:), test(kk+1).z(:)');
    distance_matrix         = dx.^2 + dy.^2 + dz.^2;
    
end    
    
    dist_lim_hard2          = (parameters.length_high).^2;
    II                      = distance_matrix > dist_lim_hard2;
    distance_matrix(II)     = inf;
    
    non_inf   = ~isinf(distance_matrix);
    num_col   = sum(non_inf,1);
    num_row   = sum(non_inf,2);
    
    row_reduced  = find(num_row ~= 0);
    col_reduced  = find(num_col ~= 0);
    
    nn     = max(length(row_reduced),length(col_reduced));
    distance_matrix_reduced = zeros(nn);
    distance_matrix_reduced(1:length(row_reduced),1:length(col_reduced)) = distance_matrix(row_reduced,col_reduced);
    d_max = max(distance_matrix_reduced(distance_matrix_reduced(:)~=Inf));
    
    edge     = distance_matrix_reduced;
    II       = distance_matrix_reduced ~= Inf;
    edge(II) = 0;
     
    [n_add] = correct_deficiency(edge);        
    nn      = nn+n_add;
    %fprintf('nn: %i\t; n_add: %i\t ; d_max: %f\n',nn,  n_add, d_max);
    %distance_matrix_reduced = ones(nn)*d_max;
    
    distance_matrix_reduced = ones(nn)*d_max;
    distance_matrix_reduced(1:length(row_reduced),1:length(col_reduced)) = distance_matrix(row_reduced,col_reduced);
    
    
end
%%
function [n_add]    = correct_deficiency(edge)

  n           = length(edge)  ;
  row         = zeros(n,1) ;  
  col         = zeros(n,1) ; 
  edge_eff    = zeros(n)   ; 
  
  for ii = 1:n
    for jj = 1:n
      if edge(ii,jj) == 0 && row(ii) == 0 && col(jj) == 0
        edge_eff(ii,jj) = 1;
        row(ii) = 1;
        col(jj) = 1;
        break;
      end
    end
  end
  
row = zeros(n,1);  

indicator_1 = 1;
while indicator_1  

      row_loc = 0;
      col_loc = 0;
      indicator_2 = 1;
      ii = 1;
      jj = 1;
      
      while indicator_2
          
          if edge(ii,jj) == 0 && row(ii) == 0 && col(jj) == 0
            row_loc = ii;
            col_loc = jj;
            indicator_2 = 0;
          end 
          
          jj = jj + 1;      
          if jj > n;
              jj = 1;
              ii = ii+1;
          end      
          if ii > n;
              indicator_2 = 0;
          end      
      end

      if row_loc == 0
        indicator_1 = 0;
      else

        edge_eff(row_loc,col_loc) = 2;

        if sum(edge_eff(row_loc,:)==1) ~= 0
        
            row(row_loc)  = 1;
            col_col2      = edge_eff(row_loc,:)==1;
            col(col_col2) = 0;
        
        else

            indicator_1 = 0;

        end            
      end
end  
  


    n_add = n - sum(col + row);


  
end
%%
function [binary_localization, real_assignment, index_global] = kuhn_munktree(distance_matrix,nn,row_reduced,col_reduced,n,m)

indicator_1  = 1;
goto         = 1;
index_global = 0;
while indicator_1
    switch goto
        case 1
            [distance_matrix,goto] = remove_min(distance_matrix);
            index_global = index_global + 1;
        case 2
            [first_zeros,row,col,goto] = search_first_zeros(distance_matrix, nn);
            index_global = index_global + 1;
        case 3
            [col,goto]         = collumn_check(first_zeros,nn);
            index_global = index_global + 1;
        case 4
            [first_zeros,row,col,target_row,target_col,goto] = zero_priming(distance_matrix,row,col,first_zeros,nn);
            index_global = index_global + 1;
        case 5
            [first_zeros,row,col,goto] = sequential_starring(first_zeros,target_row,target_col,nn);
            index_global = index_global + 1;
        case 6
            [distance_matrix,goto] = add_remove_minimal(distance_matrix,row,col,nn);
            index_global = index_global + 1;
        case 7
            indicator_1 = 0;
    end
    
    
end

n_row_red = length(row_reduced);
n_col_red = length(col_reduced);

binary_localization                          = zeros(n,m);
binary_localization(row_reduced,col_reduced) = first_zeros([1:n_row_red],[1:n_col_red]);
[value_loc, nij]                             = max( binary_localization , [] , 2 );
real_assignment                              = nij.*value_loc;


end
%%
function [distance_matrix,goto] = remove_min(distance_matrix)
  
  distance_matrix = bsxfun(@minus,distance_matrix , min(distance_matrix,[],2));
  goto = 2;
    
end
%%
function [first_zeros,row,col,goto] = search_first_zeros(distance_matrix, nn)

  row  = zeros(nn,1) ;  
  col  = zeros(1,nn) ;  
  first_zeros      = zeros(nn)   ;  
  
  for i = 1:nn
    for j = 1:nn
      if distance_matrix(i,j) == 0 && row(i,1) == 0 && col(1,j) == 0
        first_zeros(i,j) = 1;
        row(i,1) = 1;
        col(1,j) = 1;
        break;
      end
    end
  end

  goto = 3;
  row  = zeros(nn,1) ;  
  col  = zeros(1,nn) ; 
  
end
%%
function [col,goto] = collumn_check(first_zeros,nn)

  col = sum(first_zeros,1);
  if sum(col) == nn
    goto = 7;
  else
    goto = 4;
  end
  
  
  
end
%%
function [first_zeros,row,col,target_row,target_col,goto] = zero_priming(distance_matrix,row,col,first_zeros,nn)

indicator_1 = 1;
while indicator_1  

      row_loc = 0; col_loc = 0; indicator_2 = 1;
      i = 1; j = 1;
      while indicator_2
          if distance_matrix(i,j) == 0 && row(i) == 0 && col(j) == 0
            row_loc = i;
            col_loc = j;
            indicator_2 = 0;
          end      
          j = j + 1;      
          if j > nn;
              j = 1;
              i = i+1;
          end      
          if i > nn;
              indicator_2 = 0;
          end      
      end

      if row_loc == 0
        goto = 6;
        indicator_1 = 0;
        target_row = 0;
        target_col = 0;
      else
          
          first_zeros(row_loc,col_loc) = 2;
          if sum(first_zeros(row_loc,:)==1) > 0
           
            row(row_loc) = 1;            
            col_col2 = first_zeros(row_loc,:)==1;
            col(1,col_col2) = 0;
            
          else
            goto = 5;
            indicator_1 = 0;
            target_row = row_loc;
            target_col = col_loc;
          end            
      end
end


end
%%
function [first_zeros,row,col,goto] = sequential_starring(first_zeros,target_row,target_col,nn)

  indicator_1 = 1;
  i           = 1;
  II          = [1:nn]';
  while indicator_1 

    row_index =  first_zeros(:,target_col(i))==1;

    if sum(row_index) > 0
      i = i+1;
      target_row(i,1) = II(row_index);
      target_col(i,1) = target_col(i-1,1);
    else
      indicator_1 = 0;
    end
    
    if indicator_1 == 1;
        
      column_index = first_zeros(target_row(i),:)==2;
      i = i+1;
      target_row(i,1) = target_row(i-1,1);
      target_col(i,1) = II(column_index);    
    
    end   
    
  end
  

  II                          = target_row + nn * (target_col-1);
  JJ                          = II(first_zeros(II)==1);
  KK                          = II(first_zeros(II)~=1);
  first_zeros(   JJ    )      = 0;
  first_zeros(   KK    )      = 1;
  
  
  row                         = zeros(nn,1);
  col                         = zeros(1,nn);
  
  first_zeros(first_zeros==2) = 0;

goto = 3;

end
%%
function [distance_matrix,goto] = add_remove_minimal(distance_matrix,row,col,nn)

row_zeros       = row == 0;
col_zeros       = col == 0;

%row_ones        = ~row_zeros;
row_ones        =  row == 1;


row_expand      = repmat( row_zeros, [1, nn] );
col_expand      = repmat( col_zeros, [nn, 1] );

%col_expand      = repmat( col_zeros', [nn, 1] );


product         = distance_matrix.* row_expand .* col_expand;
JJ              = product ~=0;
minimal         = min(product(JJ));

distance_matrix = distance_matrix + minimal*repmat(row_ones, [1, nn]);
distance_matrix = distance_matrix - minimal*repmat(col_zeros,[nn , 1] );

goto = 4;

end
%%
function res = cost_function_distance(distance_matrix, binary_localization)

II = isinf(distance_matrix);
distance_matrix(II) = 1e10;
CC = distance_matrix.*binary_localization;
res = sum( CC(:) )  ;

end
%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                       End Hungarian Solution                          %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                      approx solution Min-Sum 1                        %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [out] = approx_min_sum_solution_1(test,parameters,i)

[log_likelihood_matrix,log_likelihood_matrix_reduced, distance_matrix, row_reduced,col_reduced, nn]  = compute_all_approx(test, i, parameters);
[n,m]                              = size(log_likelihood_matrix);
[boolean_matrix1, boolean_matrix2] = generate_boolean_approx(nn) ;
II                                 = 1:nn;
[x_A, x_B]                         = updates_as_bipartite_approx( nn, log_likelihood_matrix_reduced, parameters,boolean_matrix1, boolean_matrix2, II);
[~,~, binary_localization_raw, real_assignment, ~, state ]  = find_optimal_assigment(x_A,x_B, nn,  log_likelihood_matrix_reduced , II);
%% 
n_row_red = length(row_reduced);
n_col_red = length(col_reduced);
binary_localization                          = zeros(n,m);
binary_localization(row_reduced,col_reduced) = binary_localization_raw([1:n_row_red],[1:n_col_red]);
%% last cleaning
CC                                          = log_likelihood_matrix.*binary_localization;
tt                                          = min(CC, [],2);
JJ                                          = tt==0;
tt(JJ)                                      = NaN; 
JJ                                          = (isnan(tt))|(-tt >parameters.length_high.^2 );
binary_localization(JJ,:)                   = 0;
%%
[value_loc, nij]                            = max( binary_localization , [] , 2 );
real_assignment                             = nij.*value_loc;
cost_function_Value                         = cost_function(log_likelihood_matrix, binary_localization);  
        
II                           = 1:length(log_likelihood_matrix(:,1));    
t                            = test(i).t; 
x1                           = test(i).x;
y1                           = test(i).y; 
x2                           = test(i+1).x;
y2                           = test(i+1).y;
% out.t                     = t;
% out.x1                    = x1;
% out.y1                    = y1;
% out.x2                    = x2;
% out.y2                    = y2;
if (parameters.d == 3)
    z1                       = test(i).z;
    z2                       = test(i+1).z;
%     out.z1                   = z1;
%     out.z2                   = z2;
end

% out.x = x1;
% out.y = y1;
% if (parameters.d == 3)
%     out.z                 = z1;
% end

 JJ                           = real_assignment ~= 0;   


out.x_all = x1;
out.y_all = y1;
out.x     = x1(JJ);
out.y     = y1(JJ);

if (parameters.d == 3)
    out.z                 = z1(JJ);
    out.z_all             = z1;
end
out.t                        = t.*ones(length(out.x),1);
out.t_all                 = t.*ones(length(out.x_all),1);

out.dx                    = x2(real_assignment(JJ)) - x1(JJ);
out.dy                    = y2(real_assignment(JJ)) - y1(JJ);
if (parameters.d == 3)
    out.dz                = z2(real_assignment(JJ)) - z1(JJ);
end

if isempty(II(JJ)')
%    fprintf('Here NaN \n');
    out.index_assigned           = NaN;
    out.real_assignment          = NaN;
    out.assigment                = NaN;
else
 %   fprintf('Here Normal \n');
    out.index_assigned           = II(JJ)';
    out.real_assignment          = real_assignment;
    out.assigment                = binary_localization;
end
% out.state                    = state;
% out.cost                     = cost_function_Value;
% out.log_lik                  = log_likelihood_matrix;

end
%%
function [log_likelihood_matrix,log_likelihood_matrix_reduced, distance_matrix, row_reduced,col_reduced, nn] = compute_all_approx(test, kk, parameters)


if (parameters.d ==2)
    
    dx                      = bsxfun(@minus, test(kk).x(:), test(kk+1).x(:)');
    dy                      = bsxfun(@minus, test(kk).y(:), test(kk+1).y(:)');
    distance_matrix         = dx.^2 + dy.^2;    

else 
    
    dx                      = bsxfun(@minus, test(kk).x(:), test(kk+1).x(:)');
    dy                      = bsxfun(@minus, test(kk).y(:), test(kk+1).y(:)');
    dz                      = bsxfun(@minus, test(kk).z(:), test(kk+1).z(:)');
    distance_matrix         = dx.^2 + dy.^2 + dz.^2;
    
end    

    dist_lim_hard2          = (parameters.length_high).^2;
    II                      = distance_matrix > dist_lim_hard2;
    distance_matrix(II)     = inf;
    log_likelihood_matrix   = -distance_matrix;
    
    
    non_inf   = ~isinf(distance_matrix);
    num_col   = sum(non_inf,1);
    num_row   = sum(non_inf,2);
    
    row_reduced  = find(num_row ~= 0);
    col_reduced  = find(num_col ~= 0);
    
    nn     = max(length(row_reduced),length(col_reduced));
    distance_matrix_reduced = zeros(nn);
    distance_matrix_reduced(1:length(row_reduced),1:length(col_reduced)) = distance_matrix(row_reduced,col_reduced);
    d_max = max(distance_matrix_reduced(distance_matrix_reduced(:)~=Inf));
    
    edge     = distance_matrix_reduced;
    II       = distance_matrix_reduced ~= Inf;
    edge(II) = 0;
     
    [n_add] = correct_deficiency(edge);        
    nn      = nn+n_add;
    %fprintf('nn: %i\t; n_add: %i\t ; d_max: %f\n',nn,  n_add, d_max);
    %distance_matrix_reduced = ones(nn)*d_max;
    
    distance_matrix_reduced = ones(nn)*2*d_max + 10*rand(nn)*d_max;
    
    distance_matrix_reduced(1:length(row_reduced),1:length(col_reduced)) = distance_matrix(row_reduced,col_reduced);    
    log_likelihood_matrix_reduced = -distance_matrix_reduced;

end
%%
function [boolean_matrix1, boolean_matrix2] = generate_boolean_approx(nn) 



boolean_matrix1  = ones(nn, nn, nn);
boolean_matrix2  = ones(nn, nn, nn);
for i =1 : nn;
    boolean_matrix1(i,:,i) = NaN;
    boolean_matrix2(i,:,i) = NaN;
end;


end
%%
function [x_A, x_B] =  initialisation_approx(kind_init, nn)

switch lower(kind_init)
    
    case 'zeros'
        [x_A, x_B] = pure_0_initialize_approx(nn);
    case 'random'
        [x_A, x_B] = random_0_1_initialize_approx(nn);
    case 'normal'
        [x_A, x_B] = normal_initialize_approx(nn); 
             
end;

end
%%
function [x_A, x_B] = random_0_1_initialize_approx(nn)

    x_A = rand(nn);
    x_B = rand(nn);

end
%%
function [x_A, x_B] = pure_0_initialize_approx(nn)

    x_A = zeros(nn);
    x_B = zeros(nn);

end
%%
function [x_A, x_B] = normal_initialize_approx(nn)

    x_A = randn(nn);
    x_B = randn(nn);

end
%%
function [x_A, x_B] = updates_as_bipartite_approx( nn, log_likelihood_matrix, parameters,boolean_matrix1, boolean_matrix2, II, test, i)

number_loop         = parameters.number_stop ;                       
nb_duration_state_1 = parameters.nb_duration_state_1;
global_state        = zeros(nb_duration_state_1,1);
kind_init           = parameters.kind_init  ;
index               = 0;
[x_A_old, x_B_old]  = initialisation_approx(kind_init,nn);


  % while ( floor( sum( global_state )./nb_duration_state_1)~=1  && (index < number_loop) )
    while ( (index < number_loop) )     


         x_A_new  = reshape(  min( bsxfun(@times, boolean_matrix1, -log_likelihood_matrix  - x_B_old  ),[],1),nn,nn ) ;
         x_B_new  = reshape(  min( bsxfun(@times, boolean_matrix2, -log_likelihood_matrix' - x_A_old  ),[],1),nn,nn ) ;        


        [~,~, binary_localization, real_assigment, number_assigment, state ] = find_optimal_assigment(x_A_new,x_B_new, nn, log_likelihood_matrix , II);
        
        %cost_new                                                = cost_function(log_likelihood_matrix, binary_localization);
%         global_state(mod(index,nb_duration_state_1)+1,1) = state;
%         
%         if (mod(index,250) == 0)
%             
%             plot_assignment(test,i, parameters,real_assigment);
%             axis equal;
%             drawnow;
%             %fprintf('%i\t %i\t %i\n ', index, number_loop, state);
%             %fprintf('number assigned: %i\t; nn: %i\n',number_assigment, nn );
%             %fprintf('x_A: %f\t; x_B: %i\n',x_A_new(125,125), x_B_new(125,125) );
%             
%         end
    
        x_A_old  = x_A_new;
        x_B_old  = x_B_new;
    
        index   = index +1;
    end;
    
 x_A = x_A_new;
 x_B = x_B_new;


end
%%
function [binary_localization,real_assigment,number_assigment,nij,nji, state,p_motion]  = clean_up_approx(log_likelihood_matrix, binary_localization, parameters,x_A,x_B, ~, ~ , II, min_n_A_n_B)


CC                        = log_likelihood_matrix.*binary_localization;
tt                        = min(CC, [],2);
JJ                        = tt==0;
tt(JJ)                    = NaN; 
JJ                        = (isnan(tt))|(-tt >parameters.length_high );
binary_localization(JJ,:) = 0;
[~, nij]                  = min( -log_likelihood_matrix  - x_B , [] , 2 );
[~, nji]                  = min( -log_likelihood_matrix' - x_A , [] , 2 );
status                    = sum(binary_localization,2);
real_assigment            = nij(II).*( status ) ;
number_assigment          = sum (status);
state                     = number_assigment == min_n_A_n_B;


end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                   end approx solution Min-Sum 1                       %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                      approx solution Min-Sum 2                        %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [out] = approx_min_sum_solution_2(test,parameters,i)

[log_likelihood_matrix]                     = compute_all_approx_2(test, i, parameters);
[n_A,n_B]                                   = size(log_likelihood_matrix);
[boolean_matrix1, boolean_matrix2]          = generate_boolean_assymetry(n_A, n_B);
II                                          = 1:n_A;
[x_A, x_B]                                  = update_non_bipartite( n_A, n_B, log_likelihood_matrix, parameters,boolean_matrix1, boolean_matrix2);
[~,~, binary_localization, ~, ~, state ]    = find_optimal_assigment_non_bipartite(x_A,x_B, n_A,n_B, log_likelihood_matrix , II);

%% last cleaning
CC                                          = log_likelihood_matrix.*binary_localization;
tt                                          = min(CC, [],2);
JJ                                          = tt==0;
tt(JJ)                                      = NaN; 
JJ                                          = (isnan(tt))|(-tt >parameters.length_high.^2 );
binary_localization(JJ,:)                   = 0;
%%
[value_loc, nij]                            = max( binary_localization , [] , 2 );
real_assignment                             = nij.*value_loc;
cost_function_Value                         = cost_function(log_likelihood_matrix, binary_localization);  
%%        
II                           = 1:length(log_likelihood_matrix(:,1));  
t                            = test(i).t; 
x1                           = test(i).x;
y1                           = test(i).y; 
x2                           = test(i+1).x;
y2                           = test(i+1).y;
% out.t                     = t;
% out.x1                    = x1;
% out.y1                    = y1;
% out.x2                    = x2;
% out.y2                    = y2;
if (parameters.d == 3)
    z1                       = test(i).z;
    z2                       = test(i+1).z;
%     out.z1                   = z1;
%     out.z2                   = z2;
end

% out.x = x1;
% out.y = y1;
% if (parameters.d == 3)
%     out.z                 = z1;
% end

JJ                           = real_assignment ~= 0;   




out.x_all = x1;
out.y_all = y1;
out.x     = x1(JJ);
out.y     = y1(JJ);

if (parameters.d == 3)
    out.z                 = z1(JJ);
    out.z_all             = z1;
end
out.t                        = t.*ones(length(out.x),1);
out.t_all                 = t.*ones(length(out.x_all),1);

out.dx                    = x2(real_assignment(JJ)) - x1(JJ);
out.dy                    = y2(real_assignment(JJ)) - y1(JJ);
if (parameters.d == 3)
    out.dz                = z2(real_assignment(JJ)) - z1(JJ);
end

if isempty(II(JJ)')
%    fprintf('Here NaN \n');
    out.index_assigned           = NaN;
    out.real_assignment          = NaN;
    out.assigment                = NaN;
else
 %   fprintf('Here Normal \n');
    out.index_assigned           = II(JJ)';
    out.real_assignment          = real_assignment;
    out.assigment                = binary_localization;
end
% out.state                    = state;
% out.cost                     = cost_function_Value;
% out.log_lik                  = log_likelihood_matrix;

end
%%
function [log_likelihood_matrix] = compute_all_approx_2(test, kk, parameters)


if (parameters.d ==2)
    
    dx                      = bsxfun(@minus, test(kk).x(:), test(kk+1).x(:)');
    dy                      = bsxfun(@minus, test(kk).y(:), test(kk+1).y(:)');
    distance_matrix         = dx.^2 + dy.^2;    

else 
    
    dx                      = bsxfun(@minus, test(kk).x(:), test(kk+1).x(:)');
    dy                      = bsxfun(@minus, test(kk).y(:), test(kk+1).y(:)');
    dz                      = bsxfun(@minus, test(kk).z(:), test(kk+1).z(:)');
    distance_matrix         = dx.^2 + dy.^2 + dz.^2;
    
end

log_likelihood_matrix = -distance_matrix;

end
%%
function [boolean_matrix1, boolean_matrix2] = generate_boolean_assymetry(n_A, n_B) 

boolean_matrix1  = ones(n_A, n_B, n_A);
for i =1 : n_A;
    boolean_matrix1(i,:,i) = NaN;
end;

boolean_matrix2  = ones(n_B, n_A, n_B);
for i =1 : n_B;
    boolean_matrix2(i,:,i) = NaN;
end;



end
%%
function [x_A, x_B] = update_non_bipartite( n_A, n_B, log_likelihood_matrix, parameters,boolean_matrix1, boolean_matrix2)

kind_init              =  parameters.kind_init   ;                  
nombre_loop            =  parameters.number_stop ;
[x_A_old, x_B_old]     =  initialisation_non_bipartite(kind_init, n_A, n_B);
indice                 =  0;

while(  indice < nombre_loop )

        x_A_new  =    reshape(  min( bsxfun(@times, boolean_matrix1, -log_likelihood_matrix  - x_B_old  ),[],1),n_B,n_A );
        x_B_new  =    reshape(  min( bsxfun(@times, boolean_matrix2, -log_likelihood_matrix' - x_A_old  ),[],1),n_A,n_B );
    
        x_A_old  = x_A_new;
        x_B_old  = x_B_new;
    
        indice   = indice +1;
end;


x_A = x_A_new;
x_B = x_B_new;

end
%%
function [x_A, x_B] = initialisation_non_bipartite(kind_init, n_A, n_B)

switch lower(kind_init)
    
    case 'zeros'
        [x_A, x_B] = pure_0_initialize_non_bipartite(n_A, n_B);
    case 'random'
        [x_A, x_B] = random_0_1_initialize_non_bipartite(n_A, n_B);
    case 'normal'
        [x_A, x_B] = normal_initialize_non_bipartite(n_A, n_B); 
             
end;

end
%%
function [x_A, x_B] = random_0_1_initialize_non_bipartite(n_A, n_B)

    x_A = rand(n_B,n_A);
    x_B = rand(n_A,n_B);

end
%%
function [x_A, x_B] = pure_0_initialize_non_bipartite(n_A, n_B)

    x_A = zeros(n_B,n_A);
    x_B = zeros(n_A,n_B);

end
%%
function [x_A, x_B] = normal_initialize_non_bipartite(n_A, n_B)

    x_A = randn(n_B,n_A);
    x_B = randn(n_A,n_B);

end
%%
function [nij,nji, binary_localization, real_assignment, number_assigment, state ] = find_optimal_assigment_non_bipartite(x_A,x_B, n_A,n_B, log_likelihood_matrix , II)

[~, nij]                                           = min( -log_likelihood_matrix  - x_B , [] , 2 );
[~, nji]                                           = min( -log_likelihood_matrix' - x_A , [] , 2 );

binary_localization                                = false(n_A, n_B);
status                                             = II' ==  nji(nij(II));
min_n_A_n_B                                        = min(n_A, n_B);


binary_localization( ( nij(II) - 1 ).*n_A + II' )  = status ;
real_assignment                                     = nij(II).*( status ) ;
 number_assigment                                   = sum (status);
state                                               = number_assigment == min_n_A_n_B;

end
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                   end approx solution Min-Sum 2                       %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                                Add Stuff                              %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%
function plot_assignment(test,i, parameters, real_assignment)

    close;
    x1 = test(i).x;
    y1 = test(i).y;
    x2 = test(i+1).x;
    y2 = test(i+1).y;
    plot(x1, y1, '.k', 'MarkerSize', 15); 
    hold on;
    plot(x2, y2, '.r', 'Markersize', 15); 

    for j =1:length(real_assignment)
        if (real_assignment(j)~=0)
        plot([x1(j) x2(real_assignment(j))], [y1(j) y2(real_assignment(j))], '-g', 'LineWidth',2)
        end
    end
    


end
%%
function out = junk_solution(test,parameters,i)
    
    t                            = test(i).t;    
%     x1                           = test(i).x;
%     y1                           = test(i).y; 
%     x2                           = test(i+1).x;
%     y2                           = test(i+1).y;
    %out.t                     = t;
    
    
%     out.x1                    = x1;
%     out.y1                    = y1;
%     out.x2                    = x2;
%     out.y2                    = y2;
%     if (parameters.d == 3)
%         z1                       = test(i).z;
%         z2                       = test(i+1).z;
%         out.z1                   = z1;
%         out.z2                   = z2;
%     end


    out.x_all = [];
    out.y_all = [];
    out.x = [];
    out.y = [];
    
    if (parameters.d == 3)
        out.z                 = [];
    end
    out.t                     = [];
    out.t_all                 = [];
%     out.index_assigned           = nan;
    out.dx                       = [];
    out.dy                       = [];
    if (parameters.d == 3)
        out.dz                  = [];
        out.z_all               = [];
    end

    out.index_assigned           = [];
    out.real_assignment          = [];
    out.assigment                = [];
    
%     out.real_assignment          = nan;
%     out.assigment                = nan;
%     out.state                    = nan;
%     out.cost                     = nan;
%     out.log_lik                  = nan;

end
%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                           end  Add Stuff                              %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

