
"""
    relieff(data::Array{<:Real,2}, target::Array{<:Integer,1}, m::Signed=-1, 
                 k::Integer=5, dist_func::Function=(e1, e2) -> sum(abs.(e1 .- e2), dims=2); 
                 mode::String="k_nearest", sig::Real=1.0, f_type::String="continuous")::Array{Float64,1}

Compute feature weights using ReliefF algorithm. The mode argument specifies which type of weights update to perform and can
either have the value of "k_nearest", "diff" or "exp_rank" (see reference paper). The f_type argument specifies whether the features 
are continuous or discrete and can either have the value of "continuous" or "discrete". The sig argument is used when mode has the value
of "exp_rank".

---
# Reference:
- Marko Robnik-Šikonja and Igor Kononenko. Theoretical and empirical
analysis of ReliefF and RReliefF. Machine Learning, 53(1):23–69, Oct
2003.
"""
function relieff(data::Array{<:Real,2}, target::Array{<:Integer,1}, m::Signed=-1, 
                 k::Integer=5, dist_func::Function=(e1, e2) -> sum(abs.(e1 .- e2), dims=2); 
                 mode::String="k_nearest", sig::Real=1.0, f_type::String="continuous")::Array{Float64,1}
    
    # Check if k nearest misses and hits can be found for each class.
    # If not, reduce k.
    upper_k_lim = minimum(counts(Int64.(target))) - 1
    if k > upper_k_lim
        k = upper_k_lim
        @warn @sprintf "k reduced to %d because of insufficient number of instances." upper_k_lim
    end

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
    # Note that the number of elements in the distance vector is {n \choose 2} = n!/(2!*(n-2)!) = n*(n-1)/2.
    # Add additional element with 0.
    dists = Array{Float64}(undef, Int64((size(data, 1)^2 - size(data, 1))/2 + 1))
    dists[1] = 0  # Set first value of distances vector to 0 - accessed when i == j in square form indices.

    # Construct pairwise distances vector using vectorized distance function.
    top_ptr = 2
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
        neigh_idx = Int64.(square_to_vec(row_idxs[target .== target[idx]], col_idxs[target .== target[idx]], size(data, 1))) .+ 2
        idx_k_nearest_same = partialsortperm(dists[neigh_idx], 1:k+1)[2:end]
        
        # Get k nearest hits.
        k_nearest_same = data[target .== target[idx], :][idx_k_nearest_same, :]
       
        # Allocate matrix for storing the k nearest misses.
        k_nearest_other = Array{Float64}(undef, k * (length(keys(classes_map)) - 1), size(data, 2))

        # Go over class values not equal to class value of currently sampled sample. 
        top_ptr = 1
        @inbounds for cl = keys(classes_map)
            if cl != target[idx]
                # If class not equal to sampled example, find indices in distance vector of examples with this class.
                neigh_idx_nxt = Int64.(square_to_vec(row_idxs[target .== cl], col_idxs[target .== cl], size(data, 1))) .+ 2
                idx_k_nearest_other_nxt = partialsortperm(dists[neigh_idx_nxt], 1:k)
                k_nearest_other_nxt = data[target .== cl, :][idx_k_nearest_other_nxt, :]

                # Find k closest examples from this class.
                k_nearest_other[top_ptr:top_ptr+k-1, :] = k_nearest_other_nxt
                top_ptr += k
            end
        end

        # Get probabilities of classes not equal to class of sampled example.
        p_classes_other = p_classes[p_classes[:, 1] .!= target[idx], 2]
        
        # Compute diff sum weights for closest examples from different classes.
        p_weights = p_classes_other./(1 .- p_classes[p_classes[:, 1] .== target[idx], 2])

        # Compute diff sum weights for closest examples from different class.
        weights_mult = reshape(repeat(p_weights, inner=k), :, 1)


        ### Weights Update - K-NEAREST #######
        if mode == "k_nearest"

            if f_type == "continuous"
                # If features continuous.

                # Penalty term.
                penalty = sum(abs.(data[idx:idx, :] .- k_nearest_same)./((max_f_vals .- min_f_vals) .+ eps(Float64)), dims=1)

                # Reward term.
                reward = sum(weights_mult .* (abs.(data[idx:idx, :] .- k_nearest_other)./((max_f_vals .- min_f_vals) .+ eps(Float64))), dims=1)

                # Weights update.
                weights = weights .- penalty./(m*k) .+ reward./(m*k)

            elseif f_type == "discrete"
                # If features discrete.
                weights = weights .- sum(Int64.(data[idx:idx, :] .!= k_nearest_same), dims=1)./m .+ 
                    sum(weights_mult .* Int64.(data[idx:idx, :] .!= k_nearest_other), dims=1)./m
            else
                throw(DomainError(f_type, "f_type can only be equal to \"continuous\" or \"discrete\"."))
            end

        ### Weights Update - DIFF ############
        elseif mode == "diff"

            # Compute weights for each nearest hit
            d_vals_closest_same = 1 ./ (sum(abs.(data[idx:idx, :] .- k_nearest_same)./((max_f_vals .- min_f_vals) .+ eps(Float64)), dims=2) .+ eps(Float64))
            dist_weights_penalty = d_vals_closest_same ./ sum(d_vals_closest_same)

            # Distance weights for reward term
            d_vals_closest_other = 1 ./ (sum(abs.(data[idx:idx, :] .- k_nearest_other)./((max_f_vals .- min_f_vals) .+ eps(Float64)), dims=2) .+ eps(Float64))
            dist_weights_reward = d_vals_closest_other ./ sum(d_vals_closest_other)

            if f_type == "continuous"
                # If features continuous.
                
                # Penalty term
                penalty = sum(dist_weights_penalty .* (abs.(data[idx:idx, :] .- k_nearest_same)./((max_f_vals .- min_f_vals) .+ eps(Float64))), dims=1)
                
                # Reward term
                reward = sum(weights_mult .* (dist_weights_reward .* (abs.(data[idx:idx, :] .- k_nearest_other)./((max_f_vals .- min_f_vals) .+ eps(Float64)))), dims=1)

                # Weights update
                weights = weights .- penalty./m .+ reward./m

            elseif f_type == "discrete"
                # If features discrete.
                weights = weights .- sum(dist_weights_penalty .* Int64.(data[idx:idx, :] .!= k_nearest_same), dims=1)./m .+
                    sum(weights_mult .* dist_weights_reward .* Int64.(data[idx:idx, :] .!= k_nearest_other), dims=1)./m

            else
                throw(DomainError(f_type, "f_type can only be equal to \"continuous\" or \"discrete\"."))
            end

        ### Weights Update - EXP-RANK ########
        elseif mode == "exp_rank"
            
            # Compute weights for nearest hits.
            # NOTE: nearest hits are already sorted by distance.
            exp_rank_same = exp.(-(1:size(k_nearest_same, 1)/sig))
            closest_same_weights = exp_rank_same/sum(exp_rank_same)
            
            # Compute weights for nearest misses.
            sp = sortperm(vec(dist_func(data[idx:idx, :], k_nearest_other)))
            closest_other_ranks = collect(1:size(k_nearest_other, 1))
            closest_other_ranks[sp] = closest_other_ranks
            exp_rank_other = exp.(-(closest_other_ranks/sig))
            closest_other_weights = exp_rank_other/sum(exp_rank_other)

            if f_type == "continuous"
                # If features continuous.
            
                # Penalty term
                penalty = sum(closest_same_weights .* (abs.(data[idx:idx, :] .- k_nearest_same)./((max_f_vals .- min_f_vals) .+ eps(Float64))), dims=1)
                
                # Reward term
                reward = sum(weights_mult .* (closest_other_weights .* (abs.(data[idx:idx, :] .- k_nearest_other)./((max_f_vals .- min_f_vals) .+ eps(Float64)))), dims=1)

                # Weights update
                weights = weights .- penalty./m .+ reward./m

            elseif f_type == "discrete"
                # If features discrete.
                weights = weights .- sum(closest_same_weights .* Int64.(data[idx:idx, :] .!= k_nearest_same), dims=1)./m .+
                    sum(weights_mult .* closest_other_weights .* Int64.(data[idx:idx, :] .!= k_nearest_other), dims=1)./m

            else
                throw(DomainError(f_type, "f_type can only be equal to \"continuous\" or \"discrete\"."))
            end

        else
            throw(DomainError(f_type, "mode can only be equal to \"k_nearest\", \"diff\" or \"exp_rank\"."))
        end

        ######################################

    end
   
    # Return computed feature weights.
    return vec(weights)

end
