
"""
    get_mean_m_h_vals(data::Array{<:Real,2}, target::Array{<:Integer,1}, 
                           dist_weights::Array{<:AbstractFloat}, sig::Real; f_type::String)::Tuple{Array{<:AbstractFloat}, Array{<:AbstractFloat}}

Get mean m and mean h values (see reference, auxiliary function). The f_type argument specifies whether the 
features are continuous or discrete and can either have the value of "continuous" or "discrete".

"""
function get_mean_m_h_vals(data::Array{<:Real,2}, target::Array{<:Integer,1}, 
                           dist_weights::Array{<:AbstractFloat}, sig::Real; f_type::String)::Tuple{Array{<:AbstractFloat}, Array{<:AbstractFloat}}
    
    # Allocate matrix for storing results.
    mean_h = Array{Float64}(undef, size(data))
    mean_m = Array{Float64}(undef, size(data))

    # Go over rows of pairwise differences.
    @inbounds for idx = 1:size(data, 1)

        if f_type == "continuous"
            # If features continuous.
            # Compute m values.
            m_vals = abs.(reshape(data[idx, :], 1, length(data[idx, :])) .- data[target .!= target[idx], :])
            h_vals = abs.(reshape(data[idx, :], 1, length(data[idx, :])) .- data[(target .== target[idx]) .& (1:size(data, 1) .!= idx), :])

        elseif f_type == "discrete"
            # If features discrete.
            # Compute m values.
            m_vals = Int64.(reshape(data[idx, :], 1, length(data[idx, :])) .!= data[target .!= target[idx], :])
            h_vals = Int64.(reshape(data[idx, :], 1, length(data[idx, :])) .!= data[(target .== target[idx]) .& (1:size(data, 1) .!= idx), :])
        else
            throw(DomainError(f_type, "f_type can only be equal to \"continuous\" or \"discrete\"."))
        end

        # Compute kernel function values.
        f_m_vals = exp.(-sum(reshape(dist_weights, 1, length(dist_weights)).*m_vals, dims=2)/sig)
        f_h_vals = exp.(-sum(reshape(dist_weights, 1, length(dist_weights)).*h_vals, dims=2)/sig)

        # Compute vector of probabilities of misses being nearest misses.
        pm_vec = f_m_vals./(sum(f_m_vals) + eps(Float64))
        ph_vec = f_h_vals./(sum(f_h_vals) + eps(Float64))

        # Compute mean_m_values for each example
        mean_m[idx, :] = sum(pm_vec.*m_vals, dims=1)
        mean_h[idx, :] = sum(ph_vec.*h_vals, dims=1)
    end

    return mean_m, mean_h
end


"""
    get_gamma_vals(data::Array{<:Real,2}, target::Array{<:Integer,1}, dist_weights::Array{<:AbstractFloat}, sig::Real; f_type::String)

Get gamma values (see reference, auxiliary function). The f_type argument specifies whether the 
features are continuous or discrete and can either have the value of "continuous" or "discrete".
"""
function get_gamma_vals(data::Array{<:Real,2}, target::Array{<:Integer,1}, dist_weights::Array{<:AbstractFloat}, sig::Real; f_type::String)
    
    # Allocate array for storing results.
    po_vals = Array{Float64}(undef, size(data, 1))

    # Go over rows of distance matrix.
    @inbounds for idx = 1:size(data, 1)
        
        # Compute probability of n-th example being an outlier.
        if f_type == "continuous"
            # If features continuous.
            m_vals = abs.(reshape(data[idx, :], 1, length(data[idx, :])) .- data[target .!= target[idx], :])
            d_vals = abs.(reshape(data[idx, :], 1, length(data[idx, :])) .- data)
        elseif f_type == "discrete"
            # If features discrete.
            m_vals = Int64.(reshape(data[idx, :], 1, length(data[idx, :])) .!= data[target .!= target[idx], :])
            d_vals = Int64.(reshape(data[idx, :], 1, length(data[idx, :])) .!= data)
        else
            throw(DomainError(f_type, "f_type can only be equal to \"continuous\" or \"discrete\"."))
        end

        f_m_vals = exp.(-sum(reshape(dist_weights, 1, length(dist_weights)).*m_vals, dims=2)/sig)
        f_d_vals = exp.(-sum(reshape(dist_weights, 1, length(dist_weights)).*d_vals, dims=2)/sig)
        po_vals[idx] = sum(f_m_vals)/(sum(f_d_vals) + eps(Float64))
    end
    
    # Gamma values are probabilities of examples being inliers.
    return 1 .- po_vals
end


"""
    get_nu(gamma_vals::Array{<:AbstractFloat}, mean_m_vals::Array{<:AbstractFloat}, mean_h_vals::Array{<:AbstractFloat}, nrow::Integer)

Get nu value (see reference, auxiliary function).
"""
function get_nu(gamma_vals::Array{<:AbstractFloat}, mean_m_vals::Array{<:AbstractFloat}, mean_h_vals::Array{<:AbstractFloat}, nrow::Integer)
    return (1/nrow) .* sum(reshape(gamma_vals, length(gamma_vals), 1) .* (mean_m_vals .- mean_h_vals), dims=1)
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
                 initial_w_div::Real=-1.0; f_type::String="continuous")::Array{<:AbstractFloat}
    
    # If initial weight divisor argument has signal value of -1.0, set to I (number of features).
    if initial_w_div == -1.0
        initial_w_div = size(data, 2)
    end

    # Intialize convergence indicator and distance weights for features.
    convergence = false 
    dist_weights = ones(Float64, size(data, 2))/initial_w_div

    # Initialize iteration counter.
    iter_count = 0

    ### Main iteration loop. ###
    @inbounds while iter_count < max_iter && !convergence

        # Get gamma values and compute nu.
        gamma_vals = get_gamma_vals(data, target, dist_weights, k_width, f_type=f_type)
        
        # Get mean m and mean h vals for all examples.
        mean_m_vals, mean_h_vals = get_mean_m_h_vals(data, target, dist_weights, k_width, f_type=f_type)
        
        # Get nu vector.
        nu = get_nu(gamma_vals, mean_m_vals, mean_h_vals, size(data, 1)) 

        # Update distance weights.
        dist_weights_nxt = clamp.(nu, 0, Inf)/(LinearAlgebra.norm(clamp.(nu, 0, Inf)) + eps(Float64))

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
