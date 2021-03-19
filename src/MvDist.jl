struct MvDist{C<:CorMat}
    margins::Vector{UD}
    target_cor::C
    adjust_cor::CorMat
end
MvDist(margins::Vector{UD}, rho::CorMat{Nothing}) = MvDist(margins, rho, rho)
function MvDist(margins::Vector{UD}, rho::CorMat{Pearson})
    # rho_adjust = match_pearson
    # MvDist(margins, rho, CorMat{Nothing}(rho_adjust))
end
function MvDist(margins::Vector{UD}, rho::CorMat{C}) where {C<:Union{Spearman, Kendall}}
    rho_adjust = convert(CorMat{Pearson}, rho)
    MvDist(margins, rho, CorMat{Nothing}(rho_adjust))
end
function MvDist(margins::Vector{UD}, rho::Matrix{Float64}, C::AbstractCorrelation)
    rho_adjust = CorMat{Nothing}(cor_adjust(rho, C))
    MvDist(margins, CorMat{C}, rho_adjust)
end



function Base.show(io::IO, ::MIME"text/plain", D::MvDist)
    num_margins = size(D.margins, 1)
    v, h = displaysize(io)
    num_margins_displayable = v - 6
    first_n = num_margins_displayable ÷ 2
    last_n = num_margins_displayable - first_n
    C = typeof(D.target_cor)
    println(io, "$(num_margins)-dimensional $(typeof(D))")

    if num_margins_displayable > num_margins
        for m in D.margins
            println(io, " ", m)
        end
    else
        for i in 1:first_n
            println(io, " ", D.margins[i])
        end
        println(io, " ⋮")
        for i in num_margins-last_n:num_margins-1
            println(io, " ", D.margins[i])
        end
        print(io, " ", D.margins[end])
    end
end
