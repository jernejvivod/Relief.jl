
"""
    function get_mean_m_h_vecs(data::Array{<:Real,2}, target::Array{<:Integer,1}, dist_func::Any, 
                               sig::Real; f_type::String)::Tuple{Array{<:AbstractFloat}, Array{<:AbstractFloat}}

Get mean m and mean h values (see reference, auxiliary function). The f_type argument specifies whether the 
features are continuous or discrete and can either have the value of "continuous" or "discrete".
"""
function get_mean_m_h_vecs(data::Array{<:Real,2}, target::Array{<:Integer,1}, dist_func::Any, 
                           sig::Real; f_type::String)::Tuple{Array{<:AbstractFloat}, Array{<:AbstractFloat}}
    
    # Allocate matrices for storing results.
    mean_m = Array{Float64}(undef, size(data))
    mean_h = Array{Float64}(undef, size(data))

    # Go over samples and compute weighted mean m and h vectors.
    @inbounds for idx = 1:size(data, 1)

        if f_type == "continuous"
            # If features continuous.
            # Compute m and h matrices.
            m_vecs = abs.(data[idx:idx, :] .- data[target .!= target[idx], :])
            h_vecs = abs.(data[idx:idx, :] .- data[(1:size(data, 1) .!= idx) .& (target .== target[idx]), :])

        elseif f_type == "discrete"
            # If features discrete.
            # Compute m and h matrices.
            m_vecs = Int64.(data[idx:idx, :] .!= data[target .!= target[idx], :])
            h_vecs = Int64.(data[idx:idx, :] .!= data[(1:size(data, 1) .!= idx) .& (target .== target[idx]), :])
        else
            throw(DomainError(f_type, "f_type can only be equal to \"continuous\" or \"discrete\"."))
        end

        # Compute mean m vector for next sample.
        f_dists_other_m_i = exp.(-dist_func(data[idx:idx, :], data[target .!= target[idx], :])/sig)
        f_dists_other_sum_m_i = sum(f_dists_other_m_i)
        alpha_i = f_dists_other_m_i / (f_dists_other_sum_m_i + eps(Float64))    # Compute alpha vector.
        mean_m[idx, :] = m_vecs'*alpha_i

        # Compute mean h vector for next sample.
        f_dists_other_h_i = exp.(-dist_func(data[idx:idx, :], data[(1:size(data, 1) .!= idx) .& (target .== target[idx]), :])/sig)
        f_dists_other_sum_h_i = sum(f_dists_other_h_i)         
        beta_i = f_dists_other_h_i / (f_dists_other_sum_h_i + eps(Float64))     # Compute beta vector.
        mean_h[idx, :] = h_vecs'*beta_i
    end

    # Return results.
    return mean_m, mean_h
end


"""
    get_gamma_vals(data::Array{<:Real,2}, target::Array{<:Integer,1}, dist_func::Any, sig::Real; f_type::String)

Get gamma values (see reference, auxiliary function). The f_type argument specifies whether the 
features are continuous or discrete and can either have the value of "continuous" or "discrete".
"""
function get_gamma_vals(data::Array{<:Real,2}, target::Array{<:Integer,1}, dist_func::Any, sig::Real; f_type::String)
    
    # Allocate array for storing results.
    po_vals = Array{Float64}(undef, size(data, 1))

    # Go over samples and compute probabilities of sample being an outlier.
    @inbounds for idx = 1:size(data, 1)
        
        # Compute probability of n-th example being an outlier.
        if f_type == "continuous"
            # If features continuous.
            m_vecs = abs.(data[idx:idx, :] .- data[target .!= target[idx], :])
            d_vals = abs.(data[idx:idx, :] .- data)
        elseif f_type == "discrete"
            # If features discrete.
            m_vecs = Int64.(data[idx:idx, :] .!= data[target .!= target[idx], :])
            d_vals = Int64.(data[idx:idx, :] .!= data)
        else
            throw(DomainError(f_type, "f_type can only be equal to \"continuous\" or \"discrete\"."))
        end

        # Compute P_{o} value for next sample.
        f_dists_other_m_i = exp.(-dist_func(data[idx:idx, :], data[target .!= target[idx], :])/sig)
        f_dists_other_d_i = exp.(-dist_func(data[idx:idx, :], data[1:size(data, 1) .!= idx, :])/sig)
        po_vals[idx] = sum(f_dists_other_m_i)/(sum(f_dists_other_d_i) + eps(Float64))
    end
    
    # Return probabilities of examples being inliers.
    return permutedims(1 .- po_vals)'
end


"""
    irelief(data::Array{<:Real,2}, target::Array{<:Integer,1}, max_iter::Integer, k_width::Real, conv_condition::Real, 
                 initial_w_div::Real; f_type::String="continuous")::Array{<:AbstractFloat}

Compute feature weights using I-Relief algorithm. The f_type argument specifies whether the features are continuous or discrete 
and can either have the value of "continuous" or "discrete".

---
# Reference:
- Yijun Sun and Jian Li. Iterative RELIEF for feature weighting. In ICML
2006 - Proceedings of the 23rd International Conference on Machine
Learning, volume 2006, pages 913–920, 2006.
"""
function irelief(data::Array{<:Real,2}, target::Array{<:Integer,1}, max_iter::Integer=1000, k_width::Real=2.0, conv_condition::Real=1.0e-6, 
                 initial_w_div::Real=-1.0, dist_func::Any=(e1, e2, w) -> sum(abs.(w.*(e1 .- e2)), dims=2); f_type::String="continuous")::Array{<:AbstractFloat}
    
    # If initial weight divisor argument has signal value of -1.0, set to I (number of features).
    if initial_w_div == -1.0
        initial_w_div = size(data, 2)
    end

    # Intialize convergence indicator and distance weights for features.
    convergence = false 
    dist_weights = ones(Float64, 1, size(data, 2))/initial_w_div

    # Initialize iteration counter.
    iter_count = 0

    ### Main iteration loop. ###
    @inbounds while iter_count < max_iter && !convergence

        # Get mean m and mean h vals for all examples.
        mean_m_vecs, mean_h_vecs = get_mean_m_h_vecs(data, target, (e1, e2) -> dist_func(e1, e2, dist_weights), k_width, f_type=f_type)

        # Get gamma values and compute nu.
        gamma_vals = get_gamma_vals(data, target, (e1, e2) -> dist_func(e1, e2, dist_weights), k_width, f_type=f_type)

        # Get nu vector.
        nu = ((mean_m_vecs - mean_h_vecs)'*gamma_vals)'
        
        # Update distance weights.
        nu_clipped = clamp.(nu, 0, Inf)
        dist_weights_nxt = nu_clipped/(LinearAlgebra.norm(nu_clipped) + eps(Float64))

        # Check if convergence criterion satisfied. If not, continue with next iteration.
        if sum(abs.(dist_weights_nxt .- dist_weights)) < conv_condition
            dist_weights = dist_weights_nxt
            convergence = true
        else
            dist_weights = dist_weights_nxt
            iter_count += 1
        end 
    end

    ############################

    # Return feature ranks and last distance weights.
    return vec(dist_weights)
end
