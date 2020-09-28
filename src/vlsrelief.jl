
"""
    vlsrelief(data::Array{<:Real,2}, target::Array{<:Integer,1}, num_partitions_to_select::Integer, 
                   num_subsets::Integer, partition_size::Integer, rba::Any=Relief.relieff, 
                   f_type::String="continuous")::Array{Float64,1}

Compute feature weights using VLSRelief algorithm. The num_partitions_to_select argument specifies how many partitions
to select in each iteration. The num_subsets argument specifies the number of subset iterations to perform. The partition_size
argument specifies the size of a partition. The rba argument specifies a (partially applied) wrapped 
RBA algorithm that should accept just the data and target values.

---
# Reference:
- Margaret Eppstein and Paul Haake. Very large scale ReliefF for genome-
wide association analysis. In 2008 IEEE Symposium on Computational
Intelligence in Bioinformatics and Computational Biology, CIBCB ’08,
2008.
"""
function vlsrelief(data::Array{<:Real,2}, target::Array{<:Integer,1}, partition_size::Integer=-1,
                    num_partitions_to_select::Integer=-1, num_subsets::Integer=100; rba::Any=Relief.relieff)::Array{Float64,1}
    
    # If partition_size argument has signal value of -1, compute default value from data size.
    if partition_size == -1
        partition_size = Int64(size(data, 2)/10)
    end

    # If num_partitions_to_select argument has signal value of -1, compute default value from data size.
    if num_partitions_to_select == -1
        num_partitions_to_select = partition_size*5
    end

    # Initialize feature weights vector.
    weights = zeros(Float64, size(data, 2))

    # Get vector of feature indices.
    feat_ind = collect(1:size(data, 2))

    # Get indices of partition starting indices.
    feat_ind_start_pos = collect(1:partition_size:size(data, 2))

    # Go over subsets and compute local ReliefF scores.
    @inbounds for i = 1:num_subsets

        # Randomly select k partitions and combine them to form a subset of features of examples.
        ind_sel = [collect(el:el+partition_size-1) for el in StatsBase.sample(feat_ind_start_pos, num_partitions_to_select, replace=false)]

        # Flatten list of indices' lists.
        ind_sel_unl = Array{Int64}(undef, num_partitions_to_select*partition_size)
        ptr = 1
        @inbounds for sel = ind_sel
            ind_sel_unl[ptr:ptr+partition_size-1] = sel
            ptr += partition_size
        end
        ind_sel_unl = ind_sel_unl[ind_sel_unl .<= feat_ind[end]]
        
        # Use RBA on subset to obtain local weights.
        rba_weights = rba(data[:, ind_sel_unl], target)

        # Update weights using computed local weights.
        weights[ind_sel_unl] = vec(maximum(hcat(weights[ind_sel_unl], rba_weights), dims=2))
    end

    # Return computed feature weights.
    return weights
end
