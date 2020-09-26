
"""
    min_radius(n::Integer, data::Array{<:Real,2}, target::Array{<:Integer,1}, dist_func::Any)::Float64

Compute minimum raidus of hypersphere centered around each data sample so that each hypersphere contains
at least n samples from same class as the corresponding data sample as well as n samples from a different class.

Author: Jernej Vivod
"""
function min_radius(n::Integer, data::Array{<:Real,2}, target::Array{<:Integer,1}, dist_func::Any)::Float64

    # Allocate array for storing minimum acceptable radius for each example in dataset.
    min_r = Array{Float64}(undef, size(data, 1))

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

    # Go over examples and compute minimum acceptable radius for each example.
    @inbounds for k = 1:size(data, 1)

        # Row and column indices for querying pairwise distance vector.
        row_idxs = repeat([k - 1], size(data, 1))
        col_idxs = collect(0:size(data, 1)-1)

        # Get indices in distance vector (from square form indices).
        neigh_idx = Int64.(square_to_vec(row_idxs, col_idxs, size(data, 1)) .+ 2)
        
        # Get distance from current sample.
        dist_from_e = dists[neigh_idx[neigh_idx .!= 0]]

        # Get mask for samples from same class.
        msk = target .== target[k]
        
        # Get distances to samples from same class.
        dist_same = dist_from_e[msk]

        # Get distances to samples from different class.
        dist_diff = dist_from_e[.!msk]

        # Compute minimum radius for next sample.
        try
            min_r[k] = max(sort(dist_same)[n+1], sort(dist_diff)[n])
        catch e
            error("Insufficient examples with class $(target[k]) for given value of n (n = $(n))")
        end
    end
   
    # Return maximum of array of minimum acceptable radiuses for each example
    return maximum(min_r)
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


    # Get minimum radius needed to include n samples from same class and n samples
    # from different class for each sample.
    min_r = min_radius(min_incl, data, target, (e1,  e2) -> dist_func(e1, e2, ones(Float64, 1, size(data,2))))
    
    # Initialize distance weights.
    dist_weights = ones(Float64, size(data, 2))

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
        
        # Go over sampled samples.
        @inbounds for idx in sample_idxs

            # Get next sampled sample.
            e = data[idx, :]
            
            # Get filtered data.
            data_filt = data[1:end .!= idx, :]
            target_filt = target[1:end .!= idx] 

            # Compute hypersphere inclusions and distances to examples within the hypersphere.
            # Distances to examples from same class.
            
            dist_same_all = dist_func(data_filt[target_filt .== target[idx], :], reshape(e, 1, length(e)), reshape(dist_weights, 1, length(dist_weights)))
            sel = dist_same_all .<= min_r
            dist_same = dist_same_all[sel]
            data_same = (data_filt[target_filt .== target[idx], :])[vec(sel), :]
            
            # Distances to examples with different class.
            dist_other_all = dist_func(data_filt[target_filt .!= target[idx], :], reshape(e, 1, length(e)), reshape(dist_weights, 1, length(dist_weights)))
            sel = dist_other_all .<= min_r
            dist_other = dist_other_all[sel]
            data_other = (data_filt[target_filt .!= target[idx], :])[vec(sel), :]

            ### Weights Update ###

            w_miss = max.(0, 1 .- (dist_other.^2/min_r.^2))
            w_hit = max.(0, 1 .- (dist_same.^2/min_r.^2))

            if f_type == "continuous"
                # If features continuous.
        
                numerator1 = sum(abs.(reshape(e, 1, length(e)) .- data_other) .* w_miss, dims=1)
                denominator1 = sum(w_miss) + eps(Float64)

                numerator2 = sum(abs.(reshape(e, 1, length(e)) .- data_same) .* w_hit, dims=1)
                denominator2 = sum(w_hit) + eps(Float64)

                feature_weights .+= vec(numerator1 ./ denominator1 .- numerator2 ./ denominator2)

            elseif f_type == "discrete"
                # If features discrete.
                
                numerator1 = sum(Int64.(reshape(e, 1, length(e)) .!= data_other) .* w_miss, dims=1)
                denominator1 = sum(w_miss) + eps(Float64)

                numerator2 = sum(Int64.(reshape(e, 1, length(e)) .!= data_same) .* w_hit, dims=1)
                denominator2 = sum(w_hit) + eps(Float64)

                feature_weights .+= vec(numerator1 ./ denominator1 .- numerator2 ./ denominator2)

            else
                throw(DomainError(f_type, "f_type can only be equal to \"continuous\" or \"discrete\"."))
            end

            ######################
        
        end

        # Update distance weights by feature weights - use algorithm's own feature evaluations
        # to weight features when computing distances.
        dist_weights .+= feature_weights

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
