function fill_Z!(Z, i, end_ind, Z_)
    @inbounds Z[:,:,i:end_ind] .= Array(Z_); # TODO: Can we do this faster??
end

function retrieve_sparse_representation(data_matrix, len, projs, hp, ffts, cdl; F=float_type)
    @info "Retreiving the sparse code and the filters..."
    Z = Array{F,3}(undef, (len.C,hp.M,len.Nf));
    ZᵀDZ_sum_sum = CUDA.zeros(F, (len.cs_vlen, hp.M, 1));
    ZᵀS_sum = CUDA.zeros(F, (len.cs_vlen, hp.M, 1));

    D_final, mu = square_non_neg_params_du(cdl)
    D_cpu = Array(D_final);

    @inbounds for i = 1:hp.batch_size:len.Nf
        end_ind = i+hp.batch_size-1 ≤ len.Nf ? i+hp.batch_size-1 : len.Nf
        d_gpu = cu(data_matrix[:,:,i:end_ind]);
        Z_ = CSC(d_gpu, cdl, len, hp, projs);
        fill_Z!(Z, i, end_ind, Z_)

        zeros_Zs, zeros_Sz, rFzs, irFzs, rFsz = batch_prep_du(size(d_gpu,3), len, projs, ffts)

        DZ_sum = sum(convolution(Z_, D_final, pad=len.f_len_inc, groups=hp.M), dims=2); # DZ's sum; shape: (len.cs_vlen, 1, N)

        Zr = reverse(Z_, dims=1)
        ZᵀDZ_sum, ZᵀS = BregmanPrep(Zr, DZ_sum, d_gpu, rFzs, rFsz, irFzs, zeros_Zs, zeros_Sz)

        
        ZᵀDZ_sum_sum .+= sum(ZᵀDZ_sum, dims=3)
        ZᵀS_sum .+= sum(ZᵀS, dims=3)
    end

    ZᵀDZ_sum_sum = Array(ZᵀDZ_sum_sum); ZᵀS_sum = Array(ZᵀS_sum);
    projs_mapfrange_cpu = Array(projs.mapfrange);
    D_grad   = reshape(batched_mul(projs_mapfrange_cpu,  ZᵀS_sum + ZᵀDZ_sum_sum), (size(projs.mapfrange,1), 1, hp.M));
    Breg_num = reshape(D_cpu .* exp.(-mu[1] .* D_grad), (4, hp.filter_len, 1, hp.M));
    D_final_cpu = reshape((Breg_num ./ sum(Breg_num,dims=1)), (hp.f_len, 1, hp.M))
    return D_final_cpu, Z
end

function feed_dataloader(data_matrix, hp)
    return Flux.DataLoader(data_matrix, batchsize=hp.batch_size, shuffle=true)
end

function training_prep(M, f, batch_size, K_c, gamma_1, pool_mask_len, pool_stride, data_matrix, learning_rate)
    hp              = HyperParam(batch_size, f, f*4, M, K_c, 1, gamma_1, pool_mask_len, pool_stride);
    data_load       = feed_dataloader(data_matrix, hp);
    len             = length_info(data_load, data_matrix, hp)
    projs           = projectors(hp, len);
    ffts            = cuda_fft_plans(len, hp);
    cdl             = CDL(hp, 1, K_c+1);
    ps              = Flux.params(cdl);
    opt             = Flux.AdaBelief(learning_rate)
    return hp, data_load, len, projs, ffts, cdl, ps, opt
end

function train_basic(data;
                     M             = 160, 
                     num_epochs    = 25, 
                     f             = 8, 
                     K_c           = 3, 
                     gamma_1       = 0.005, 
                     batch_size    = 48,
                     pool_mask_len = 40,
                     pool_stride   = 4,
                     learning_rate = 0.0003, 
                     save_sparse_representation = false,
                     save_loc = nothing
                     )
    
    N = size(data.data_matrix,3);   
    batch_size = N < batch_size ? N-1 : batch_size;
    # @info "# filters: $M , filter length: $f, epochs: $num_epochs, # coding layers: $K_c, γ: $gamma_1, batch size: $batch_size, learning_rate: $learning_rate"
    N % batch_size == 0 && (batch_size = batch_size + 1)

    hp, data_load, len, projs, ffts, cdl, ps, opt =
        training_prep(M, f, batch_size, K_c, gamma_1, 
            pool_mask_len, pool_stride, data.data_matrix, learning_rate)

    for iter = 1:num_epochs
        for d in data_load
            d = d |> gpu;
            gs = gradient(ps) do 
                CDLforward(d, cdl, len, hp, projs, ffts)
            end
            Flux.Optimise.update!(opt, ps, gs) # update parameters                
        end
        # just to show the training loss
        if iter % 5 == 0 
            @info "$iter epoch done."
            # loss_value = CDLforward(first(data_load) |> gpu, cdl, len, hp, projs, ffts);
            # @info "epoch: $iter, batch loss: $loss_value"
        end
    end
    D, Z = retrieve_sparse_representation(data.data_matrix, len, projs, hp, ffts, cdl)

    # TODO: in the case where Z is big, save it into HDF5 format

    if save_sparse_representation
        where2save = nothing; # the sparse_rep directory in save_loc directory
        if isnothing(save_loc) 
            where2save = joinpath(pwd(), "sparse_rep"); mkdir("sparse_rep")
        else
            where2save = joinpath(save_loc, "sparse_rep"); 
            !isdir(where2save) && mkpath(where2save)        
        end
        @save joinpath(where2save, "filters.jld2") D
        @save joinpath(where2save, "sparse_code.jld2") Z
        @save joinpath(where2save, "data.jld2") data
        @save joinpath(where2save, "filterlen.jld2") f
    else
        return Z, D
    end
end



