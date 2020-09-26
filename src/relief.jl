
"""
    relief(data::Array{<:Real,2}, target::Array{<:Integer,1}, m::Signed=-1, 
                dist_func::Any=(e1, e2) -> sum(abs.(e1 .- e2), dims=2); f_type::String="continuous")::Array{Float64,1}

Compute feature weights using Relief algorithm. The f_type argument specifies whether the features are continuous or discrete 
and can either have the value of "continuous" or "discrete".

---
# Reference:
- Kenji Kira and Larry A. Rendell. The feature selection problem: Tra-
ditional methods and a new algorithm. In Proceedings of the Tenth
National Conference on Artificial Intelligence, AAAI’92, pages 129–134.
AAAI Press, 1992.
"""
function relief(data::Array{<:Real,2}, target::Array{<:Integer,1}, m::Signed=-1, 
                dist_func::Any=(e1, e2) -> sum(abs.(e1 .- e2), dims=2); f_type::String="continuous")::Array{Float64,1}

    # Initialize feature weights vector.
    weights = zeros(Float64, size(data, 2))

    # Compute vectors of maximum and minimum feature values.
    max_f_vals = vec(maximum(data, dims=1))
    min_f_vals = vec(minimum(data, dims=1))
    
    # Sample m examples without replacement. If m has signal value -1, use all examples.
    sample_idxs = StatsBase.sample(1:size(data, 1), if (m==-1) size(data,1) else m end, replace=false)
    if (m == -1) m = size(data, 1) end # If m has signal value -1, set m to total number of examples.

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
        
        # Get indices of neighbours with same class in distances vector and find nearest hit.
        neigh_idx_hit = Int64.(square_to_vec(row_idxs[target .== target[idx]], col_idxs[target .== target[idx]], size(data, 1))) .+ 2
        idx_nearest_hit = partialsortperm(dists[neigh_idx_hit], 1:2)[2:end]
        nearest_hit = vec(data[target .== target[idx], :][idx_nearest_hit, :])

        # Get indices of neighbours with different class in distances vector and find nearest miss.
        neigh_idx_miss = Int64.(square_to_vec(row_idxs[target .!= target[idx]], col_idxs[target .!= target[idx]], size(data, 1))) .+ 2
        idx_nearest_miss = argmin(dists[neigh_idx_miss])
        nearest_miss = vec(data[target .!= target[idx], :][idx_nearest_miss, :])
        
        ### Weights Update ###
       
        if f_type == "continuous"
            # If features continuous.
            weights = weights .- (abs.(data[idx, :] .- nearest_hit)./(max_f_vals .- min_f_vals .+ eps(Float64)))./m .+
                (abs.(data[idx, :] .- nearest_miss)./(max_f_vals .- min_f_vals .+ eps(Float64)))./m 
        elseif f_type == "discrete"
            # If features discrete.
            weights = weights .- Int64.(data[idx, :] .!= nearest_hit)./m .+ Int64.(data[idx, :] .!= nearest_miss)./m
        else
            throw(DomainError(f_type, "f_type can only be equal to \"continuous\" or \"discrete\"."))
        end

        ######################

    end

    # Return computed feature weights.
    return weights
end 
