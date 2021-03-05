
"""
    dm_vals(e::Array{<:Real,2}, closest::Array{<:Real,2}, max_f_vals::Array{<:Real,2}, min_f_vals::Array{<:Real,2})

Compute dm values for each feature (see reference, auxiliary function).
"""
function dm_vals(e::Array{<:Real,2}, closest::Array{<:Real,2}, max_f_vals::Array{<:Real,2}, min_f_vals::Array{<:Real,2})

    # Allocate matrix for results.
	results =  Array{Float64, 2}(undef, size(closest))

    # Compute diff values and dm values.
	diff_vals = abs.(e .- closest)./(max_f_vals .- min_f_vals .+ eps(Float64))
	@inbounds for i = 1:length(e)
		results[:,i] = Statistics.mean(diff_vals[:, 1:end .!= i], dims=2)
	end

    # Return dm value for each feature.
	return results, diff_vals
end


"""
    reliefmss(data::Array{<:Real,2}, target::Array{<:Integer,1}, m::Signed=-1, 
                   k::Integer=10, dist_func::Any=(e1, e2) -> sum(abs.(e1 .- e2), dims=2); 
                   f_type::String="continuous")

Compute feature weights using ReliefMSS algorithm. The f_type argument specifies whether the features are continuous or discrete 
and can either have the value of "continuous" or "discrete".

---
# Reference:
- Salim Chikhi and Sadek Benhammada. ReliefMSS: A variation on a
feature ranking ReliefF algorithm. IJBIDM, 4:375–390, 2009.
"""
function reliefmss(data::Array{<:Real,2}, target::Array{<:Integer,1}, m::Signed=-1, 
                   k::Integer=10, dist_func::Any=(e1, e2) -> sum(abs.(e1 .- e2), dims=2); 
                   f_type::String="continuous")
    
    # Initialize feature weights vector.
    weights = zeros(Float64, 1, size(data, 2))

    # Compute vectors of maximum and minimum feature values.
    max_f_vals = maximum(data, dims=1)
    min_f_vals = minimum(data, dims=1)
    
    # Sample m examples without replacement.
    sample_idxs = StatsBase.sample(1:size(data, 1), if (m==-1) size(data,1) else m end, replace=false)
    if (m == -1) m = size(data, 1) end
    
    # Compute probabilities of class values in training set.
    classes_map = countmap(target)
    num_samples = length(target)
    p_classes = Array{Float64}(undef, length(keys(classes_map)), 2)
    @inbounds for (idx, p) = enumerate(classes_map)  # Compute matrix representation of probabilities.
        p_classes[idx, :] = [p[1], p[2]/num_samples]
    end

    # Compute pairwise distances between samples (vector form).
    dists = Array{Float64}(undef, Int64((size(data, 1)^2 - size(data, 1))/2))

    # Construct pairwise distances vector using vectorized distance function.
    top_ptr = 1
    @inbounds for idx = 1:size(data,1)-1
        upper_lim = top_ptr + size(data, 1) - idx - 1
        dists[top_ptr:upper_lim] = dist_func(data[idx:idx, :], data[idx+1:end, :])
        top_ptr = upper_lim + 1
    end
    
    # Go over sampled indices.
    @inbounds for idx = sample_idxs

        # Row and column indices for querying pairwise distance vector.
        row_idxs = repeat([idx - 1], size(data, 1))
        col_idxs = collect(0:size(data, 1)-1)

        # Get indices in distance vector (from square form indices).
        neigh_idx = square_to_vec(row_idxs[(target .== target[idx]) .& (1:length(target) .!= idx)], 
                                  col_idxs[(target .== target[idx]) .& (1:length(target) .!= idx)], size(data, 1)) .+ 1
        idx_k_nearest_same = partialsortperm(dists[neigh_idx], 1:k)
        
        # Get k nearest hits.
        k_nearest_same = data[(target .== target[idx]) .& (1:length(target) .!= idx), :][idx_k_nearest_same, :]
       
        # Allocate matrix for storing the k nearest misses.
        k_nearest_other = Array{Float64}(undef, k * (length(keys(classes_map)) - 1), size(data, 2))

        # Go over class values not equal to class value of currently sampled sample. 
        top_ptr = 1
        @inbounds for cl = keys(classes_map)
            if cl != target[idx]
                # If class not equal to sampled example, find indices in distance vector of examples with this class.
                neigh_idx_nxt = Int64.(square_to_vec(row_idxs[target .== cl], col_idxs[target .== cl], size(data, 1))) .+ 1
                idx_k_nearest_other_nxt = partialsortperm(dists[neigh_idx_nxt], 1:k)
                
                # Get k closest examples from this class and store in matrix.
                k_nearest_other_nxt = data[target .== cl, :][idx_k_nearest_other_nxt, :]
                k_nearest_other[top_ptr:top_ptr+k-1, :] = k_nearest_other_nxt
                top_ptr += k
            end
        end


        ### MARKING CONSIDERED FEATURES ###
        
        # Compute DM values and DIFF values for each feature of each nearest hit and nearest miss.
        dm_vals_same, diff_vals_same = dm_vals(data[idx:idx, :], k_nearest_same, max_f_vals, min_f_vals)
        dm_vals_other, diff_vals_other = dm_vals(data[idx:idx, :], k_nearest_other, max_f_vals, min_f_vals)

        # Compute masks for considered features of nearest hits and nearest misses.
        features_msk_same = diff_vals_same .> dm_vals_same
        features_msk_other = diff_vals_other .> dm_vals_other

        ###################################


        # Get probabilities of classes not equal to class of sampled example.
        p_classes_other = p_classes[p_classes[:, 1] .!= target[idx], 2]
        
        # Compute diff sum weights for closest examples from different classes.
        p_weights = p_classes_other./(1 .- p_classes[p_classes[:, 1] .== target[idx], 2])
        weights_mult = reshape(repeat(p_weights, inner=k), :, 1)


        ### Weights Update ###
        #
        if f_type == "continuous"
            # If features continuous.

            # Penalty term
            penalty = (abs.(data[idx:idx, :] .- k_nearest_same)./(max_f_vals .- min_f_vals .+ eps(Float64))) .- dm_vals_same
            penalty[.!features_msk_same] .= 0.0
            penalty = sum(penalty, dims=1)

            # Reward term
            reward = weights_mult .* (abs.(data[idx:idx, :] .- k_nearest_other)./(max_f_vals .- min_f_vals .+ eps(Float64))) - dm_vals_other
            reward[.!features_msk_other] .= 0.0
            reward = sum(reward, dims=1)

            # Weights update
            weights = weights .- penalty./(m*k) .+ reward./(m*k)

        elseif f_type == "discrete"
            # If features discrete.
            
            # Penalty term
            penalty = Int64.(data[idx:idx, :] .!= k_nearest_same) .- dm_vals_same
            penalty[.!features_msk_same] .= 0.0
            penalty = sum(penalty, dims=1)

            # Reward term
            reward = weights_mult .* Int64.(data[idx:idx, :] .!= k_nearest_other) - dm_vals_other
            reward[.!features_msk_other] .= 0.0
            reward = sum(reward, dims=1)

            # Weights update
            weights = weights .- penalty./(m*k) .+ reward./(m*k)

        else
            throw(DomainError(f_type, "f_type can only be equal to \"continuous\" or \"discrete\"."))
        end

        ######################
    end

    # Return computed feature weights.
    return vec(weights)
end
