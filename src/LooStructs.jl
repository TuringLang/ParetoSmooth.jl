
using AxisKeys
using PrettyTables
export PsisLoo, AbstractLoo, AbstractLooMethod, PsisLooMethod

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


function Base.show(io::IO, ::MIME"text/plain", loo_object::PsisLoo)
    _throw_pareto_k_warnings(loo_object.pointwise(:pareto_k))
    table = loo_object.estimates
    pretty_table(table;
        compact_printing=false,
        header=table.estimate,
        row_names=table.criterion,
        formatters = ft_printf("%5.2f"),
        alignment=:r
    )
end
