module DeepUnfoldedCDLMotif

# Write your package code here.
using CUDA, Flux, Zygote, FFTW, LinearAlgebra, Random, JLD2

include("constants.jl")
include("custom_Zygote_rules.jl")
include("utils.jl")
include("model.jl")
include("opt.jl")
include("train.jl")

end
