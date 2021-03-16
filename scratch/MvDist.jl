struct MvDist{C<:CorMat}
    margins::Vector{UD}
    target_cor::C
    adjust_cor::CorMat{T, S, Pearson} where {T,S}
end
function MvDist(margins::Vector{UD}, rho::CorMat)
    adj_cor = convert(CorMat{eltype(rho), AbstractMatrix{eltype(rho)}, Pearson}, rho)
    return MvDist(margins, rho, adj_cor)
end

function Base.show(io::IO, ::MIME"text/plain", D::MvDist)
    num_margins = size(D.margins, 1)
    v, h = displaysize(io)
    num_margins_displayable = v - 6
    first_n = num_margins_displayable ÷ 2
    last_n = num_margins_displayable - first_n
    C = typeof(cor_type(D.target_cor))
    println(io, "$(num_margins)-marginal MvDist with {$(C)} target correlation")

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
