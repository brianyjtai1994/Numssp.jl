# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# Some Functions for Plotting via Matplotlib
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

export set_rcParams!, set_ticklabels!, save_pdf

function set_rcParams!(rc::F, rcParams::D; fontsize::Int=14) where {F<:Function, D<:AbstractDict}
    rc("text", usetex=true)
    rc("font", family="serif")

    rcParams["font.size"] = fontsize
    rcParams["legend.framealpha"] = 0
    rcParams["text.latex.preamble"] = ["\\usepackage{siunitx}", "\\usepackage{mhchem}"]
end

function set_ticklabels!(axis, s::T, e::T, ss::T, dir::Symbol, pad::Bool=true) where T<:Real
    ticks = collect(s:ss:e)
    if pad
        label = Vector{String}(undef, length(ticks) + 1)
        label[1] = ""
    else
        label = Vector{String}(undef, length(ticks))
    end

    tick2label!(label, ticks, pad)
    
    dir == :x ? axis[:set_xticklabels](label) :
    dir == :y ? axis[:set_yticklabels](label) :
    dir == :z ? axis[:set_zticklabels](label) :
    error("Invalid direction!")
end

function tick2label!(label::Vector{String}, ticks::Vector{T}, pad::Bool) where T<:Int
    if pad
        @inbounds @simd for i in eachindex(ticks)
            label[i+1] = string(ticks[i])
        end
    else
        @inbounds @simd for i in eachindex(ticks)
            label[i] = string(ticks[i])
        end
    end
end

function tick2label!(label::Vector{String}, ticks::Vector{T}, pad::Bool) where T<:Float64
    if pad
        @inbounds @simd for i in eachindex(ticks)
            label[i+1] = @sprintf "%.1f" ticks[i]
        end
    else
        @inbounds @simd for i in eachindex(ticks)
            label[i] = @sprintf "%.1f" ticks[i]
        end
    end
end

save_pdf(f::String, fig) = fig[:savefig](f, format="pdf", dpi=600, bbox_inches="tight")
