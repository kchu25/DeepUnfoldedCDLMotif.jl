function batch_prep_csc(this_batch_size, len, projs)
    if this_batch_size == len.N
        z_mask   = projs.z_mask_n 
        return z_mask
    elseif this_batch_size == len.Ns
        z_mask   = projs.z_mask_s 
        return z_mask
    elseif this_batch_size == 1
        z_mask   = projs.z_mask_1 
        return z_mask
    else
        error("Batch size not matched");
    end
end

function batch_prep_du(this_batch_size, len, projs, ffts)
    if this_batch_size == len.N
        zeros_Zs = projs.zeros_Zs_n 
        zeros_Sz = projs.zeros_Sz_n 
        rFzs     =      ffts.rFzs_n 
        irFzs    =     ffts.irFzs_n
        rFsz     =      ffts.rFsz_n  
        return zeros_Zs, zeros_Sz, rFzs, irFzs, rFsz
    elseif this_batch_size == len.Ns
        zeros_Zs = projs.zeros_Zs_s 
        zeros_Sz = projs.zeros_Sz_s 
        rFzs     =      ffts.rFzs_s 
        irFzs    =     ffts.irFzs_s
        rFsz     =      ffts.rFsz_s  
        return zeros_Zs, zeros_Sz, rFzs, irFzs, rFsz
    elseif this_batch_size == 1
        zeros_Zs = projs.zeros_Zs_1 
        zeros_Sz = projs.zeros_Sz_1 
        rFzs     =      ffts.rFzs_1 
        irFzs    =     ffts.irFzs_1
        rFsz     =      ffts.rFsz_1  
        return zeros_Zs, zeros_Sz, rFzs, irFzs, rFsz
    else
        error("Batch size not matched");
    end
end

function square_non_neg_params_csc(cdl)
    D_init1 = cdl.D1 .^ 2
    D_init2 = cdl.D2 .^ 2
    eta     = cdl.eta .^2
    lambda  = cdl.lambda .^2
    return D_init1, D_init2, eta, lambda    
end

function square_non_neg_params_du(cdl)
    D_final = cdl.Dfinal .^ 2
    mu      = cdl.mu .^ 2
    return D_final, mu    
end

function initial_stage(D_init, S, z_mask, eta, lambda) 
    DᵀS = convolution(S, D_init, pad=0, flipped=true);
    DᵀS = (eta[1] .* DᵀS) .- (eta[1] * lambda[1]); # shifted
    return Flux.NNlib.relu.(z_mask .* DᵀS), DᵀS;
end

function SparseCoding(Z, DᵀS, D_init, z_mask, eta, lambda, k_c, len, hp)
    DZ     = sum(convolution(Z, D_init, pad=len.f_len_inc, groups=hp.M), dims=2);
    DᵀDZ   = convolution(DZ, D_init, pad=0, flipped=true);
    Z_grad = (Z .+ eta[k_c+1] .* (DᵀS + DᵀDZ)) .- (eta[k_c+1]*lambda[k_c+1]); # shift
    return Flux.NNlib.relu.(z_mask .* Z_grad);   
end

function BregmanPrep(Zr, DZ_sum, S, rFzs, rFsz, irFzs, zeros_Zs, zeros_Sz)            
    ZᵀDZ_sum    = irfft_ZᵀS(
                    (rFzs*vcat(Zr, zeros_Zs)) .* (rFsz*vcat(DZ_sum, zeros_Sz)), 
                        irFzs, rFzs);  
    ZᵀS         = irfft_ZᵀS(
                    (rFzs*vcat(Zr, zeros_Zs)) .* (rFsz*vcat(S, zeros_Sz)), 
                        irFzs, rFzs);     
    return ZᵀDZ_sum, ZᵀS
end

function BregmanUpdate(ZᵀDZ_sum_sum, ZᵀS_sum, D_final, mu, projs, hp)
    D_grad   = reshape(
                batched_mul(projs.mapfrange,  ZᵀS_sum + ZᵀDZ_sum_sum), (size(projs.mapfrange,1), 1, hp.M));
    Breg_num = reshape(D_final .* exp.(-mu[1] .* D_grad), (4, hp.filter_len, 1, hp.M));    
    return reshape((Breg_num ./ sum(Breg_num,dims=1)), (hp.f_len, 1, hp.M));  
end

# convolutional sparse coding (for small batches)
function CSC(S, cdl, len, hp, projs)
    z_mask = batch_prep_csc(size(S,3), len, projs)    
    D_init1, D_init2, eta, lambda = square_non_neg_params_csc(cdl)
    Z, DᵀS = initial_stage(D_init1, S, z_mask, eta, lambda)
    for k_c = 1:hp.K_c
        Z = SparseCoding(Z, DᵀS, D_init2, z_mask, eta, lambda, k_c, len, hp)   
    end
    return Z
end

# dictionart update
function DU(Z, S, cdl, len, hp, projs, ffts)
    zeros_Zs, zeros_Sz, rFzs, irFzs, rFsz = batch_prep_du(size(S,3), len, projs, ffts)
    D_final, mu = square_non_neg_params_du(cdl)

    DZ_sum = sum(convolution(Z, D_final, pad=len.f_len_inc, groups=hp.M), dims=2); # DZ's sum; shape: (len.cs_vlen, 1, N)

    Zr = reverse(Z, dims=1)
    ZᵀDZ_sum, ZᵀS = BregmanPrep(Zr, DZ_sum, S, rFzs, rFsz, irFzs, zeros_Zs, zeros_Sz)

    ZᵀDZ_sum_sum  = sum(ZᵀDZ_sum, dims=3); # ZᵀDZ_sum's sum; shape: (len.cs_vlen, 1, hp.M)
    ZᵀS_sum  = sum(ZᵀS, dims=3)

    D_ =  BregmanUpdate(ZᵀDZ_sum_sum, ZᵀS_sum, D_final, mu, projs, hp)
    return D_
end

# calculate the loss of the forward pass
function network_loss(Z, D_, S, len, hp, projs)
    DZ = sum(convolution(Z, D_, pad=len.f_len_inc, groups=hp.M), dims=2)
    D__ = reshape(D_, (hp.f_len, hp.M))
    DᵀD = D__'*D__;
    (1.0/len.N)*sum((DZ - S).^2) +  hp.gamma_1*sum((DᵀD - DᵀD .* projs.qI).^2) 
end

# forward propagation (for small batches)
function CDLforward(S, cdl, len, hp, projs, ffts; get_parameters=false)
    Z = CSC(S, cdl, len, hp, projs)
    D_ = DU(Z, S, cdl, len, hp, projs, ffts)       
    if get_parameters
        return D_, Z
    else
        return network_loss(Z, D_, S, len, hp, projs)
    end
end

# get the small batch from the data-set partition
function get_S_this_batch(S, i, len)
    S_this_batch = nothing
    N_now = nothing

    if i+len.N-1 ≤ len.Nf        
        S_this_batch = @view S[:,:,i:i+len.N-1]; 
        N_now = len.N;
    elseif i+len.Ns-1 ≤ len.Nf    
        S_this_batch = @view S[:,:,i:i+len.Ns-1]; 
        N_now = len.Ns;
    else   
        S_this_batch = reshape(view(S, :,:, i), (len.L,1,1));  # TODO check if this work
        N_now = 1;
    end
    return S_this_batch, N_now
end

# do this to retrieve the full sparse code of the whole data-set
function CSC_full(S, cdl, len, hp, projs)    
    Z = Array{float_type,3}(undef, (len.C,hp.M,len.Nf))
    i = 1; 
    while i ≤ len.Nf    
        S_this_batch, N_now = get_S_this_batch(S,i,len)
        ns, ne = i, min(i+N_now-1,len.Nf)
        Z[:,:,ns:ne] = Array(CSC(S_this_batch, cdl, len, hp, projs)) # TODO check how long this takes
        i += N_now;
    end
    return Z
end