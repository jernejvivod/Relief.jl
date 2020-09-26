
"""
    multisurf(data::Array{<:Real,2}, target::Array{<:Integer,1}, 
                       dist_func::Any=(e1, e2) -> sum(abs.(e1 .- e2), dims=2); 
                       f_type::String="continuous")::Array{Float64,1}

Compute feature weights using MultiSURF algorithm. The f_type argument specifies whether the features are continuous or discrete 
and can either have the value of "continuous" or "discrete".

---
# Reference:
- Ryan Urbanowicz, Randal Olson, Peter Schmitt, Melissa Meeker, and
Jason Moore. Benchmarking Relief-based feature selection methods for
bioinformatics data mining. Journal of Biomedical Informatics, 85, 2018.
"""
function multisurf(data::Array{<:Real,2}, target::Array{<:Integer,1}, 
                       dist_func::Any=(e1, e2) -> sum(abs.(e1 .- e2), dims=2); 
                       f_type::String="continuous")::Array{Float64,1}

    # Initialize feature weights vector.
    weights = zeros(Float64, 1, size(data, 2))

    # Compute vectors of maximum and minimum feature values.
    max_f_vals = maximum(data, dims=1)
    min_f_vals = minimum(data, dims=1)

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
    

    # Go over training samples.
    @inbounds for idx = 1:size(data, 1)
        
        # Row and column indices for querying pairwise distance vector.
        row_idxs = repeat([idx - 1], size(data, 1))
        col_idxs = collect(0:size(data, 1)-1)

        # Get indices in distance vector (from square form indices).
        neigh_idx = Int64.(square_to_vec(row_idxs, col_idxs, size(data, 1)) .+ 2)

        # Query distances to neighbours to get masks for both zones.
        dists_neighbours = dists[neigh_idx]
        mu = Statistics.mean(dists_neighbours[1:length(dists_neighbours) .!= idx])
        sig = Statistics.std(dists_neighbours[1:length(dists_neighbours) .!= idx])
        thresh_near = mu - sig/2.0
        msk_near = (dists_neighbours .< thresh_near) .& (1:length(dists_neighbours) .!= idx)
        
        # Get class values of miss neighbours.
        miss_classes_near = target[msk_near .& (target .!= target[idx])]
        
        # Get masks for considered regions.
        hit_neigh_mask_near = msk_near .& (target .== target[idx])
        miss_neigh_mask_near = msk_near .& (target .!= target[idx])

        # Ignore samples that have no hits or misses in radius.
        if sum(hit_neigh_mask_near) == 0 || sum(miss_neigh_mask_near) == 0
            continue
        end
       
        # Compute weights for near misses and compute weighting vector.
        weights_mult = Array{Float64}(undef, length(miss_classes_near))  # Allocate weights multiplier vector.
        cm = countmap(miss_classes_near)  # Count unique values.
        u = collect(keys(cm))
        c = collect(values(cm)) 
        neighbour_weights = c ./ length(miss_classes_near)  # Compute misses' weights.
        @inbounds for (i, val) = enumerate(u)               # Build multiplier vector.
            find_res = findall(miss_classes_near .== val)
            weights_mult[find_res] .= neighbour_weights[i]
        end


        ### Weights Update ###

        if f_type == "continuous"
            # If features continuous.
        
            # Penalty term
            penalty = sum(abs.(data[idx:idx, :] .- data[hit_neigh_mask_near, :])./((max_f_vals .- min_f_vals) .+ eps(Float64)), dims=1)

            # Reward term
            reward = sum(weights_mult .* (abs.(data[idx:idx, :] .- data[miss_neigh_mask_near, :])./((max_f_vals .- min_f_vals .+ eps(Float64)))), dims=1)

            # Weights update
            weights = weights .- penalty./(size(data, 1)*size(data[hit_neigh_mask_near, :], 1) + eps(Float64)) .+ 
                reward./(size(data, 1)*size(data[miss_neigh_mask_near,:], 1) + eps(Float64))

        elseif f_type == "discrete"
            # If features discrete.
            
            # Penalty term
            penalty = sum(Int64.(data[idx:idx, :] .!= data[hit_neigh_mask_near, :]), dims=1)

            # Reward term
            reward = sum(weights_mult .* (Int64.(data[idx:idx, :] .!= data[miss_neigh_mask_near, :])), dims=1)

            # Weights update
            weights = weights .- penalty./(size(data, 1)*size(data[hit_neigh_mask_near, :], 1) + eps(Float64)) .+ 
                reward./(size(data, 1)*size(data[miss_neigh_mask_near,:], 1) + eps(Float64))

        else
            throw(DomainError(f_type, "f_type can only be equal to \"continuous\" or \"discrete\"."))
        end

        #####################


    end

    # Return computed feature weights.
    return vec(weights)

end
