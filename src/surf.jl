
"""
    surf(data::Array{<:Real,2}, target::Array{<:Integer,1}, m::Signed=-1, 
         dist_func::Any=(e1, e2) -> sum(abs.(e1 .- e2), dims=2); 
         f_type::String="continuous")::Array{Float64,1}

Compute feature weights using SURF algorithm. The f_type argument specifies whether the features are continuous or discrete 
and can either have the value of "continuous" or "discrete".

---
# Reference:
- Casey S. Greene, Nadia M. Penrod, Jeff Kiralis, and Jason H. Moore.
Spatially uniform ReliefF (SURF) for computationally-efficient filtering
of gene-gene interactions. BioData mining, 2(1):5–5, Sep 2009.
"""
function surf(data::Array{<:Real,2}, target::Array{<:Integer,1}, m::Signed=-1, 
              dist_func::Any=(e1, e2) -> sum(abs.(e1 .- e2), dims=2); 
              f_type::String="continuous")::Array{Float64,1}

    # Initialize feature weights vector.
    weights = zeros(Float64, 1, size(data, 2))

    # Compute vectors of maximum and minimum feature values.
    max_f_vals = maximum(data, dims=1)
    min_f_vals = minimum(data, dims=1)
    
    # Sample m samples without replacement. If m has signal value -1, use all samples.
    sample_idxs = StatsBase.sample(1:size(data, 1), if (m==-1) size(data,1) else m end, replace=false)
    if (m == -1) m = size(data, 1) end # If m has signal value -1, set m to total number of examples.
    
    # Compute pairwise distances between samples (vector form).
    dists = Array{Float64}(undef, Int64((size(data, 1)^2 - size(data, 1))/2 + 1))

    dists[1] = 0  # Set first value of distances vector to 0 - accessed when i == j in square form indices.

    # Construct pairwise distances vector using vectorized distance function.
    top_ptr = 2
    @inbounds for idx = 1:size(data,1)-1
        upper_lim = top_ptr + size(data, 1) - idx - 1
        dists[top_ptr:upper_lim] = dist_func(data[idx:idx, :], data[idx+1:end, :])
        top_ptr = upper_lim + 1
    end

    # Get mean distance between all samples.
    mean_dist = Statistics.mean(dists[2:end])

    # Go over sampled indices.
    @inbounds for idx = sample_idxs
        
        # Row and column indices for querying pairwise distance vector.
        row_idxs = repeat([idx - 1], size(data, 1))
        col_idxs = collect(0:size(data, 1)-1)

        # Get indices in distance vector (from square form indices).
        neigh_idx = square_to_vec(row_idxs, col_idxs, size(data, 1)) .+ 2
        
        # Query distances to neighbours and get mask for neighbours that fall within
        # hypersphere with radius mean_dist.
        neigh_mask = dists[neigh_idx[neigh_idx .!= 1]] .<= mean_dist
        insert!(neigh_mask, idx, 0)

        # Get masks for nearest hits and nearest misses.
        hit_neigh_mask = neigh_mask .& (target .== target[idx])
        miss_neigh_mask = neigh_mask .& (target .!= target[idx])
        
        # Ignore samples that have no hits or misses in radius.
        if sum(hit_neigh_mask) == 0 || sum(miss_neigh_mask) == 0
            continue
        end

        # Compute probability weights for misses in considered region.
        miss_classes = target[miss_neigh_mask]

        # Compute weights for nearest misses and compute weighting vector.
        weights_mult = Array{Float64}(undef, length(miss_classes))  # Allocate weights multiplier vector.
        cm = countmap(miss_classes)                                 # Count unique values.
        u = collect(keys(cm))
        c = collect(values(cm)) 
        neighbour_weights = c ./ length(miss_classes)       # Compute misses' weights.
        @inbounds for (w, val) = zip(neighbour_weights, u)  # Build multiplier vector.
            find_res = findall(miss_classes .== val)
            weights_mult[find_res] .= w
        end

        ### Weights Update ###
        
        if f_type == "continuous"
            # If features continuous.
        
            # Penalty term
            penalty = sum(abs.(data[idx:idx, :] .- data[hit_neigh_mask, :])./(max_f_vals .- min_f_vals .+ eps(Float64)), dims=1)

            # Reward term
            reward = sum(weights_mult .* abs.(data[idx:idx, :] .- data[miss_neigh_mask, :])./(max_f_vals .- min_f_vals .+ eps(Float64)), dims=1)

            # Weights update
            weights = weights .- penalty ./ (size(data, 1)*sum(hit_neigh_mask) + eps(Float64)) .+ 
            reward ./ (size(data, 1)*sum(miss_neigh_mask) + eps(Float64))

        elseif f_type == "discrete"
            # If features discrete.
            
            # Penalty term
            penalty = sum(Int64.(data[idx:idx, :] .!= data[hit_neigh_mask, :]), dims=1)

            # Reward term
            reward = sum(weights_mult .* Int64.(data[idx:idx, :] .!= data[miss_neigh_mask, :]), dims=1)

            # Weights update
            weights = weights .- penalty ./ (size(data, 1)*sum(hit_neigh_mask) + eps(Float64)) .+ 
            reward ./ (size(data, 1)*sum(miss_neigh_mask) + eps(Float64))

        else
            throw(DomainError(f_type, "f_type can only be equal to \"continuous\" or \"discrete\"."))
        end

        #####################

    end

    # Return computed feature weights.
    return vec(weights)
end
