
"""
    boostedsurf(data::Array{<:Real,2}, target::Array{<:Integer,1}, m::Signed=-1, 
                phi::Integer=3, dist_func::Any=(e1, e2, w) -> sum(w.*abs.(e1 .- e2), dims=2); 
                f_type::String="continuous")::Array{Float64,1}

Compute feature weights using BoostedSURF algorithm.

---
# Reference:
- Gediminas Bertasius, Delaney Granizo-MacKenzie, Ryan J. Urba-
nowicz, and Jason H. Moore. Boosted spatially uniform ReliefF al-
gorithm for genome-wide genetic analysis. Hanover, NH 03755, USA,
2013. Dartmouth College.

"""
function boostedsurf(data::Array{<:Real,2}, target::Array{<:Integer,1}, m::Signed=-1, 
                     phi::Integer=3, dist_func::Any=(e1, e2, w) -> sum(w.*abs.(e1 .- e2), dims=2); 
                     f_type::String="continuous")::Array{Float64,1}
    
    # Initialize feature weights vector.
    weights = zeros(Float64, 1, size(data, 2))

    # Initialize distance weights.
    dist_weights = ones(Float64, 1, size(data, 2))

    # Compute vectors of maximum and minimum feature values.
    max_f_vals = maximum(data, dims=1)
    min_f_vals = minimum(data, dims=1)
    
    # Sample m samples without replacement. If m has signal value -1, use all samples.
    sample_idxs = StatsBase.sample(1:size(data, 1), if (m==-1) size(data,1) else m end, replace=false)
    if (m == -1) m = size(data, 1) end # If m has signal value -1, set m to total number of examples.

    # Compute pairwise distances between samples (vector form).
    dists = Array{Float64}(undef, Int64((size(data, 1)^2 - size(data, 1))/2 + 1))
    dists[1] = 0  # Set first value of distances vector to 0 - accessed when i == j in square form indices.

    # Go over sampled indices.
    @inbounds for idx = sample_idxs
       
        # If idx mod phi equals 0, recompute distance metric weights by feature weights.
        if mod(idx-1, phi) == 0
            # Construct pairwise distances vector using vectorized distance function.
            dist_weights = maximum([dist_weights; ones(Float64, 1, length(dist_weights))])
            top_ptr = 2
            @inbounds for idx = 1:size(data,1)-1
                upper_lim = top_ptr + size(data, 1) - idx - 1
                dists[top_ptr:upper_lim] = dist_func(data[idx:idx, :], data[idx+1:end, :], dist_weights)
                top_ptr = upper_lim + 1
            end
        end
        
        # Row and column indices for querying pairwise distance vector.
        row_idxs = repeat([idx - 1], size(data, 1))
        col_idxs = collect(0:size(data, 1)-1)

        # Get indices in distance vector (from square form indices).
        neigh_idx = square_to_vec(row_idxs, col_idxs, size(data, 1)) .+ 2

        # Query distances to neighbors to get masks for both zones.
        dists_neighbors = dists[neigh_idx[neigh_idx .!= 1]]
        mu = Statistics.mean(dists_neighbors)
        sig = Statistics.std(dists_neighbors)
        thresh_near = mu - sig/2.0
        thresh_far = mu + sig/2.0
        neigh_mask_near = dists_neighbors .< thresh_near  
        neigh_mask_far = dists_neighbors .> thresh_far   
        insert!(neigh_mask_near, idx, 0)
        insert!(neigh_mask_far, idx, 0)
        
        # Get class values of miss neighbors.
        miss_classes_near = target[neigh_mask_near .& (target .!= target[idx])]
        miss_classes_far = target[neigh_mask_far .& (target .!= target[idx])]
        
        # Get masks for considered regions.
        hit_neigh_mask_near = neigh_mask_near .& (target .== target[idx])
        hit_neigh_mask_far = neigh_mask_far .& (target .== target[idx])
        miss_neigh_mask_near = neigh_mask_near .& (target .!= target[idx])
        miss_neigh_mask_far = neigh_mask_far .& (target .!= target[idx])

        # Ignore samples that have no hits or misses in radius.
        if sum(hit_neigh_mask_near) == 0 || sum(miss_neigh_mask_near) == 0 || 
            sum(hit_neigh_mask_far) == 0 || sum(miss_neigh_mask_far) == 0
            continue
        end

       
        # Compute weights for near misses and compute weighting vector.
        weights_mult1 = Array{Float64}(undef, length(miss_classes_near))  # Allocate weights multiplier vector.
        cm = countmap(miss_classes_near)  # Count unique values.
        u = collect(keys(cm))
        c = collect(values(cm)) 
        neighbor_weights = c ./ length(miss_classes_near)  # Compute misses' weights.

        @inbounds for (w, val) = zip(neighbor_weights, u)  # Build multiplier vector.
            find_res = findall(miss_classes_near .== val)
            weights_mult1[find_res] .= w
        end

        # Compute weights for far misses and compute weighting vector.
        weights_mult2 = Array{Float64}(undef, length(miss_classes_far))  # Allocate weights multiplier vector.
        cm = countmap(miss_classes_far)  # Count unique values.
        u = collect(keys(cm))
        c = collect(values(cm)) 
        neighbor_weights = c ./ length(miss_classes_far)   # Compute misses' weights.

        @inbounds for (w, val) = zip(neighbor_weights, u)  # Build multiplier vector.
            find_res = findall(miss_classes_far .== val)
            weights_mult2[find_res] .= w
        end


        ### Weights Update ###
        
        if f_type == "continuous"
            # If features continuous.
        
            # Penalty term for near neighbors.
            penalty_near = sum(abs.(data[idx:idx, :] .- data[hit_neigh_mask_near, :]) ./ (max_f_vals .- min_f_vals .+ eps(Float64)), dims=1)

            # Reward term for near neighbors.
            reward_near = sum(weights_mult1 .* abs.(data[idx:idx, :] .- data[miss_neigh_mask_near, :]) ./ (max_f_vals .- min_f_vals .+ eps(Float64)), dims=1)

            # Weights values for near neighbors.
            weights_near = weights .- penalty_near ./ (size(data, 1)*sum(hit_neigh_mask_near) + eps(Float64)) .+ 
                reward_near ./ (size(data, 1)*sum(miss_neigh_mask_near) + eps(Float64))


            # Penalty term for far neighbors.
            penalty_far = sum(abs.(data[idx:idx, :] .- data[hit_neigh_mask_far, :]) ./ (max_f_vals .- min_f_vals .+ eps(Float64)), dims=1)

            # Reward term for far neighbors.
            reward_far = sum(weights_mult2 .* abs.(data[idx:idx, :] .- data[miss_neigh_mask_far, :]) ./ (max_f_vals .- min_f_vals .+ eps(Float64)), dims=1)

            # Weights values for far neighbors.
            weights_far = weights .- penalty_far ./ (size(data, 1)*sum(hit_neigh_mask_far) + eps(Float64)) .+ 
                reward_far ./ (size(data, 1)*sum(miss_neigh_mask_far) + eps(Float64))

            # Update feature weights. 
            weights = weights_near - (weights_far - weights)

        elseif f_type == "discrete"
            # If features discrete.

            # Penalty term for near neighbors.
            penalty_near = sum(Int64.(data[idx:idx, :] .!= data[hit_neigh_mask_near, :]), dims=1)

            # Reward term for near neighbors.
            reward_near = sum(weights_mult1 .* Int64.(data[idx:idx, :] .!= data[miss_neigh_mask_near, :]), dims=1)

            # Weights values for near neighbors.
            weights_near = weights .- penalty_near ./ (size(data, 1)*sum(hit_neigh_mask_near) + eps(Float64)) .+ 
                reward_near ./ (size(data, 1)*sum(miss_neigh_mask_near) + eps(Float64))


            # Penalty term for far neighbors.
            penalty_far = sum(Int64.(data[idx:idx, :] .!= data[hit_neigh_mask_far, :]), dims=1)

            # Reward term for far neighbors.
            reward_far = sum(weights_mult2 .* Int64.(data[idx:idx, :] .!= data[miss_neigh_mask_far, :]), dims=1)

            # Weights values for far neighbors.
            weights_far = weights .- penalty_far ./ (size(data, 1)*sum(hit_neigh_mask_far) + eps(Float64)) .+ 
                reward_far ./ (size(data, 1)*sum(miss_neigh_mask_far) + eps(Float64))

            # Update feature weights. 
            weights = weights_near - (weights_far - weights)
            
        else
            throw(DomainError(f_type, "f_type can only be equal to \"continuous\" or \"discrete\"."))
        end
        
        ######################


    end

    # Return computed feature weights.
    return vec(weights)

end
