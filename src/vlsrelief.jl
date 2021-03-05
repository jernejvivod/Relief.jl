
"""
    vlsrelief(data::Array{<:Real,2}, target::Array{<:Integer,1}, subset_size::Integer=-1, 
                   num_subsets::Integer=100; rba::Any=relieff)::Array{Float64,1}

Compute feature weights using VLSRelief algorithm. The num_partitions_to_select argument specifies how many partitions
to select in each iteration. The num_subsets argument specifies the number of subset iterations to perform. The subset_size
argument specifies the size of a partition. The rba argument specifies a (partially applied) wrapped 
RBA algorithm that should accept just the data and target values.

---
# Reference:
- Margaret Eppstein and Paul Haake. Very large scale ReliefF for genome-
wide association analysis. In 2008 IEEE Symposium on Computational
Intelligence in Bioinformatics and Computational Biology, CIBCB ’08,
2008.
"""
function vlsrelief(data::Array{<:Real,2}, target::Array{<:Integer,1}, subset_size::Integer=-1, 
                   num_subsets::Integer=100; rba::Any=relieff)::Array{Float64,1}
    
    # If subset_size argument has signal value of -1, compute default value from data size.
    if subset_size == -1
        subset_size = Int64(ceil(size(data, 2) / 10))
    elseif subset_size > size(data, 2)  # If specified subset size greater than number of features, throw error.
        throw(DomainError(subset_size, "Feature subset size cannot exceed the number of features in the dataset."))
    end

    # Initialize feature weights vector.
    weights = zeros(Float64, size(data, 2))
    
    # Go over feature subsets.
    @inbounds for i = 1:num_subsets

        # Sample features.
        sample_idxs = StatsBase.sample(1:size(data, 2), subset_size, replace=false)

        # Use RBA on subset to obtain local weights.
        rba_weights = rba(data[:, sample_idxs], target)

        # Update weights using computed local weights.
        weights[sample_idxs] = vec(maximum(hcat(weights[sample_idxs], rba_weights), dims=2))
    end

    # Return computed feature weights.
    return weights
end
