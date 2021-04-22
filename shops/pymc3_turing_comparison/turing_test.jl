using Distributed

using CSV
using DataFrames

using StatsFuns: logistic

using Turing

using StatsPlots

addprocs(4)

@everywhere using CSV
@everywhere using DataFrames
@everywhere using StatsFuns: logistic
@everywhere using Turing

@everywhere function get_data(
        path::String="data//breast_cancer.csv",
        y_label::String="diagnosis"
    )
    df = DataFrame(CSV.File(path))
    y = df[!, y_label]
    X = convert(Matrix, select!(df, Not(:y_label)))
    X = (X .- mean(X, dims=1)) ./ std(X, dims=1)
    return X, y

end

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

@everywhere function profiler(
        X::Matrix{Float64},
        y::Vector{Int64},
        max_iters::Int64=10
    )
    times = []
    for iteration in 1:max_iters

        time = @elapsed begin
            traces = sample(
                logistic_regression(X, y),
                NUTS(1000, 0.90),
                MCMCDistributed(),
                1000,
                4
            )
        end
        append!(times, time)

    end
    return times

end


@everywhere function main()
    X, y = get_data()
    times = profiler(X, y)
    println(times)
end

main()
