using StructArrays

export PsisLoo, Loo, AbstractLoo, AbstractLooMethod, PsisLooMethod


struct LooPoint{F<:AbstractFloat}
    estimate::F
    mcse::F
    p_eff::F
    pareto_k::F
end


abstract type AbstractLoo end


struct PsisLoo{
    F<:AbstractFloat,
    AF<:AbstractArray{F},
    VF<:AbstractVector{F},
    I<:Integer,
    VI<:AbstractVector{I},
} <: AbstractLoo
    estimates::Dict{String,F}
    pointwise::StructArray{LooPoint{F}}
    psis_object::Psis{F,AF,VF,I,VI}
end


abstract type AbstractLooMethod end

struct PsisLooMethod <: AbstractLooMethod end