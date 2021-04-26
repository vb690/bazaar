using Distributed
using StatsPlots

addprocs(4)

@everywhere using CSV
@everywhere using DataFrames
@everywhere using StatsFuns: logistic
@everywhere using Turing

"""
    get_data(path, :y_label)

Returns data for the logistic model split by X, y structure. 
Retruns included breast cancer data by default.

# Arguments:
- `path::String="data/breast_cancer.csv"` path to csv file. 
- `y_label::Symbol=:diagnosis` y label for target/dependent variable

# Returns:
- `X` Matrix containing X data/exogenous variables.
- `y` Vector containing y label data/endogenous variables. 
"""
@everywhere function get_data(
        path::String="data/breast_cancer.csv",
        y_label::Symbol=:diagnosis
    )
    df = DataFrame(CSV.File(path))
    y = df[!, y_label]
    X = convert(Matrix, select!(df, Not(y_label)))
    X = (X .- mean(X, dims=1)) ./ std(X, dims=1)
    return X, y
end

"""
    logistic_regression(x::Matrix{Float64}, y::Vector{Int64})

Creates a logistic model to be used as an input argument in a sampler. 

# Arguments:
- `X::Matrix{Float64}`: Exogenous input data X in matrix format. 
- `y::Vector{Int64}`: Endogenous/dependent variables vector.
"""
@everywhere @model function logistic_regression(
        X::Matrix{Float64},
        y::Vector{Int64}
    )
    intercept ~ Normal(0, sqrt(25))

    nfeatures = size(X, 2)
    n = size(X, 1)
    slopes ~ MvNormal(nfeatures, sqrt(25))
    p = intercept .+ X * slopes
    for i = 1:n

        y[i] ~ Bernoulli(logistic(p[i]))

    end
end

"""
    profiler(X::Matrix{Float64}, y::Vector{Int64}, max_iters::Int64=10)

# Arguments:
- `X::Matrix{Float64}`: Exogenous variables
- `y::Matrix{Int64}`: Endogenous variables 
- `max_iters::Int64`: The specified number of iterations to execute the sampler for logistic regression.

# Returns:
- `times::Vector{Any}`: Execution times for each iteration.
- `trace::Chains{Any}`: The trace generated in the final sample. 
"""
@everywhere function profiler(
        X::Matrix{Float64},
        y::Vector{Int64},
        max_iters::Int64=10
    )
    times = []
    for iteration in 1:max_iters

        time = @elapsed global trace = sample(
                logistic_regression(X, y),
                NUTS(1000, 0.90),
                MCMCDistributed(),
                1000,
                4
            )
        append!(times, time)
    end
    return times, trace
end

@everywhere function main()
    X, y = get_data()
    times, traces = profiler(X, y)
    println(times)
    return times, trace
end

times, trace = main()
