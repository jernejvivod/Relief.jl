
"""
    swrfstar(data::Array{<:Real,2}, target::Array{<:Integer, 1}, m::Signed=-1, 
                  dist_func::Any=(e1, e2) -> sum(abs.(e1 .- e2), dims=2); 
                  f_type::String="continuous")::Array{Float64,1}

Compute feature weights using SWRF* algorithm. The f_type argument specifies whether the features 
are continuous or discrete and can either have the value of "continuous" or "discrete".

---
# Reference:
- Matthew E. Stokes and Shyam Visweswaran. 
Application of a spatially-weighted Relief algorithm for ranking genetic predictors of disease.
BioData mining, 5(1):20–20, Dec 2012. 23198930[pmid].
"""
function swrfstar(data::Array{<:Real,2}, target::Array{<:Integer, 1}, m::Signed=-1, 
                  dist_func::Any=(e1, e2) -> sum(abs.(e1 .- e2), dims=2); 
                  f_type::String="continuous")::Array{Float64,1}

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

    # Go over sampled examples' indices.
    @inbounds for idx = sample_idxs
        
        # Get samples from same class and different classes.
        samples_same_class = data[(target .== target[idx]) .& (1:length(target) .!= idx), :]
        samples_other_class = data[target .!= target[idx], :]

        # Compute distances to examples with same class value.
        distances_same = dist_func(data[idx:idx, :], samples_same_class)

        # Get class values of examples with different class value.
        target_other = target[target .!= target[idx]]

        # Compute t and u parameter values.
        t_same = Statistics.mean(distances_same)
        u_same = Statistics.std(distances_same)

        # Compute distances to examples with different class value.
        distances_other = dist_func(data[idx:idx, :], samples_other_class)

        # Compute t and u parameter values.
        t_other = Statistics.mean(distances_other)
        u_other = Statistics.std(distances_other)


        # Compute weights for examples from same class.
        neigh_weights_same = 2.0 ./ (1 .+ exp.(-(t_same.-distances_same)/(u_same/4.0 + eps(Float64))) .+ eps(Float64)) .- 1
        
        # Compute weights for examples from different class.
        neigh_weights_other = 2.0 ./ (1 .+ exp.(-(t_other.-distances_other)/(u_other/4.0 + eps(Float64))) .+ eps(Float64)) .- 1


        # Get probabilities of classes not equal to class of sampled example.
        p_classes_other = p_classes[p_classes[:, 1] .!= target[idx], 2]

        # Get other classes and compute diff sum weights for examples from different classes.
        classes_other = p_classes[p_classes[:, 1] .!= target[idx], 1]
        p_weights = p_classes_other./(1 .- p_classes[p_classes[:, 1] .== target[idx], 2] .+ eps(Float64))

        # Map weights vector of classes of samples with different class values to construct weights vector.
        weights_map = hcat(classes_other, p_weights)
        weights_mult = [weights_map[findall(weights_map[:, 1] .== t), 2][1] for t in target_other]
        
        
        ### Weights Update ###
        
        if f_type == "continuous"
            # If features continuous.

            # Penalty term
            penalty = sum(neigh_weights_same.*(abs.(data[idx:idx, :] .- samples_same_class)./(max_f_vals .- min_f_vals .+ eps(Float64))), dims=1)

            # Reward term
            reward = sum(neigh_weights_other.*(weights_mult .* (abs.(data[idx:idx, :] .- samples_other_class)./(max_f_vals .- min_f_vals .+ eps(Float64)))), dims=1)

            # Weights update
            weights = weights .- penalty./(m*size(samples_same_class, 1) + eps(Float64)) .+ reward./(m*size(samples_other_class, 1) + eps(Float64))

        elseif f_type == "discrete"
            # If features discrete.

            # Penalty term
            penalty = sum(neigh_weights_same.*Int64.(data[idx:idx, :] .!= samples_same_class), dims=1)

            # Reward term
            reward = sum(neigh_weights_other.*weights_mult .* Int64.(data[idx:idx, :] .!= samples_other_class), dims=1)

            # Weights update
            weights = weights .- penalty./(m*size(samples_same_class, 1) + eps(Float64)) .+ reward./(m*size(samples_other_class, 1) + eps(Float64))

        else
            throw(DomainError(f_type, "f_type can only be equal to \"continuous\" or \"discrete\"."))
        end

        ######################


    end
    
    # Return computed feature weights.
    return vec(weights)

end
