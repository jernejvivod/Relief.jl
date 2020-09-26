
"""
    turf(data::Array{<:Real,2}, target::Array{<:Integer,1}, num_it::Integer=10; 
              rba::Any=Relief.relieff)::Array{Float64,1}

Compute feature weights using TuRF algorithm. The rba argument specifies a (partially applied) wrapped 
RBA algorithm that should accept just the data and target values.

---
# Reference:
- Jason H. Moore and Bill C. White. Tuning ReliefF for genome-wide
genetic analysis. In Elena Marchiori, Jason H. Moore, and Jagath C.
Rajapakse, editors, Evolutionary Computation,Machine Learning and
Data Mining in Bioinformatics, pages 166–175. Springer, 2007.
"""
function turf(data::Array{<:Real,2}, target::Array{<:Integer,1}, num_it::Integer=10; 
              rba::Any=Relief.relieff)::Array{Float64,1}

    # Indices of features weighted by the final weights in the original data matrix.
    sel_final = collect(1:size(data,2))

    # Initialize feature weights and rank vectors.
    weights = zeros(Float64, 1, size(data, 2))
    rank = Array{Int64}(undef, size(data, 2))

    # set data_filtered equal to initial data.
    data_filtered = data

    # Flag to stop iterating if number of features to be removed becomes
    # larger than number of features left in dataset.
    stop_iterating = false

    # Compute value to add to local ranking to get global ranking.
    rank_add_val = size(data, 2)
    
    # Declare rba_weights and ind_rm in scope.
    rba_weights = nothing
    ind_rm = nothing
    
    # iteration loop
    it_idx = 0
    @inbounds while it_idx < num_it && !stop_iterating
        it_idx += 1

        # Fit RBA to get feature weights and compute feature ranks.
        rba_weights = rba(data_filtered, target)
        sp = sortperm(rba_weights, rev=true)
        rank_weights = collect(1:length(rba_weights))
        rank_weights[sp] = rank_weights

        # number of features with lowest weights to remove in each iteration (num_iterations/a).
        num_to_remove = Int64(ceil(Float64(num_it)/Float64(size(data_filtered, 2))))
        
        # If trying to remove more features than present in dataset, remove remaining features and stop iterating.
        if num_to_remove >= size(data_filtered, 2)
            num_to_remove = size(data_filtered, 2)
            stop_iterating = true
            @warn @sprintf "Iteration stopped due to all features being removed."
        end

        ### Remove num_it/a features with lowest weights. ###

        sel = rank_weights .<= (length(rank_weights) - num_to_remove)  # Get mask for removed features.
        ind_sel = findall(sel)                                         # Get indices of kept features.
        ind_rm = findall(.!sel)                                        # Get indices of removed features.
        ind_rm_original = sel_final[ind_rm]                            # Get indices of removed features in original data matrix.
        weights[ind_rm_original] = rba_weights[ind_rm]                 # Add weights of discarded features to weights vector.

        sp = sortperm(rba_weights[ind_rm], rev=true)                   # Get local ranking of removed features. 
        rank_rm = collect(1:length(rba_weights[ind_rm]))
        rank_rm[sp] = rank_rm

        rank_add_val -= num_to_remove                                  # Adjust value that converts local ranking to global ranking.
        rank[ind_rm_original] = rank_rm .+ rank_add_val                # Add value to get global ranking of removed features.
        sel_final = sel_final[ind_sel]                                 # Filter set of final selection indices.
        data_filtered = data_filtered[:, sel]                          # Filter data matrix.

        #####################################################
    end


    # Get and return final rankings and weights.
    weights_final = deleteat!(rba_weights, ind_rm)
    weights[sel_final] = weights_final

    # Return computed feature weights and rank.
    return vec(weights) #, rank

end
