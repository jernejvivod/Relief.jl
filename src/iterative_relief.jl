
"""
    min_radius(n::Integer, idx::Integer, data::Array{<:Real,2}, target::Array{<:Integer,1}, dist_func::Any)::Float64

Compute minimum raidus of hypersphere centered around each data sample so that each hypersphere contains
at least n samples from same class as the corresponding data sample as well as n samples from a different class.

Author: Jernej Vivod
"""
function min_radius(n::Integer, e_idx::Integer, data::Array{<:Real,2}, target::Array{<:Integer,1}, dist_func::Any)::Float64

    # Get masks for samples with same and different class.
    hit_neigh_mask = target .== target[e_idx]
    miss_neigh_mask = .!hit_neigh_mask
    
    # Get distances to samples from same and different classes.
    dists_same = dist_func(data[e_idx:e_idx, :], data[hit_neigh_mask, :])
    dists_other = dist_func(data[e_idx:e_idx, :], data[miss_neigh_mask, :])
    
    # Return computed radius.
    return max(sort(dists_same, dims=1)[n+1], sort(dists_other, dims=1)[n])
end


"""
    iterative_relief(data::Array{<:Real,2}, target::Array{<:Integer,1}, m::Integer=-1, min_incl::Integer=3, 
                          max_iter::Integer=100, dist_func::Any=(e1, e2, w) -> sum(abs.(w.*(e1 .- e2)), dims=2);
                          f_type::String="continuous")::Array{Float64,1}

Compute feature weights using Iterative Relief algorithm.

---
# Reference:
- Bruce Draper, Carol Kaito, and Jose Bins. Iterative Relief. Proceedings
CVPR, IEEE Computer Society Conference on Computer Vision and
Pattern Recognition., 6:62 – 62, 2003.
"""
function iterative_relief(data::Array{<:Real,2}, target::Array{<:Integer,1}, m::Integer=-1, min_incl::Integer=3, 
                          max_iter::Integer=100, dist_func::Any=(e1, e2, w) -> sum(abs.(w.*(e1 .- e2)), dims=2);
                          f_type::String="continuous")::Array{Float64,1}

    # Initialize distance weights.
    dist_weights = ones(Float64, 1, size(data, 2))

    # Initialize iteration counter, convergence indicator and
    # Array for storing feature weights from previous iteration.
    iter_count = 0
    convergence = false
    feature_weights_prev = zeros(Float64, size(data, 2))
    
    # Iterate until reached maximum iterations or convergence.
    @inbounds while iter_count < max_iter && !convergence
        
        # Increment iteration counter.
        iter_count += 1   

        # Reset feature weights to zero and sample samples.
        feature_weights = zeros(Float64, size(data, 2))
        sample_idxs = StatsBase.sample(1:size(data, 1), if (m==-1) size(data,1) else m end, replace=false)

        # Set m if currently set to signal value -1.
        if (m == -1) m = size(data, 1) end
        
        # Go over sampled indices.
        @inbounds for idx in sample_idxs

            # Get next sampled sample.
            e = data[idx:idx, :]

            # Get minimum radius needed to include n samples from same class and n samples from different class.
            min_r = min_radius(min_incl, idx, data, target, (e1, e2) -> dist_func(e1, e2, dist_weights))
            
            # Compute hypersphere inclusions and distances to examples within the hypersphere.
            # Distances to examples from same class.
            
            dist_same_all = dist_func(data[(target .== target[idx]) .& (1:length(target) .!= idx), :], e, dist_weights)
            sel = dist_same_all .<= min_r
            dist_same = dist_same_all[sel]
            data_same = data[(target .== target[idx]) .& (1:length(target) .!= idx), :][vec(sel), :]
            
            # Distances to examples with different class.
            dist_other_all = dist_func(data[target .!= target[idx], :], e, dist_weights)
            sel = dist_other_all .<= min_r
            dist_other = dist_other_all[sel]
            data_other = (data[target .!= target[idx], :])[vec(sel), :]

            ### Weights Update ###

            w_miss = max.(0, 1 .- (dist_other.^2/min_r.^2))
            w_hit = max.(0, 1 .- (dist_same.^2/min_r.^2))

            if f_type == "continuous"
                # If features continuous.
        
                numerator1 = sum(abs.(e .- data_other) .* w_miss, dims=1)
                denominator1 = sum(w_miss) + eps(Float64)

                numerator2 = sum(abs.(e .- data_same) .* w_hit, dims=1)
                denominator2 = sum(w_hit) + eps(Float64)

                feature_weights .+= vec(numerator1 ./ denominator1 .- numerator2 ./ denominator2)

            elseif f_type == "discrete"
                # If features discrete.
                
                numerator1 = sum(Int64.(e .!= data_other) .* w_miss, dims=1)
                denominator1 = sum(w_miss) + eps(Float64)

                numerator2 = sum(Int64.(e .!= data_same) .* w_hit, dims=1)
                denominator2 = sum(w_hit) + eps(Float64)

                feature_weights .+= vec(numerator1 ./ denominator1 .- numerator2 ./ denominator2)

            else
                throw(DomainError(f_type, "f_type can only be equal to \"continuous\" or \"discrete\"."))
            end

            ######################
        
        end

        # Update distance weights by feature weights - use algorithm's own feature evaluations
        # to weight features when computing distances.
        dist_weights += permutedims(feature_weights)

        # Check convergence.
        if sum(abs.(feature_weights .- feature_weights_prev)) < 1.0e-3
            convergence = true
        end
        
        # Set current feature weights as previous feature weights.
        feature_weights_prev = feature_weights
    end
    
    # Return computed feature weights.
    return vec(dist_weights)

end
