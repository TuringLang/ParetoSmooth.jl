
using AxisKeys

export PsisLoo, Loo, AbstractLoo, AbstractLooMethod, PsisLooMethod

abstract type AbstractLoo end
struct PsisLoo{
    F<:AbstractFloat,
    AF<:AbstractArray{F},
    VF<:AbstractVector{F},
    I<:Integer,
    VI<:AbstractVector{I},
} <: AbstractLoo
    estimates::KeyedArray
    pointwise::KeyedArray
    psis_object::Psis{F,AF,VF,I,VI}
end


abstract type AbstractLooMethod end

struct PsisLooMethod <: AbstractLooMethod end