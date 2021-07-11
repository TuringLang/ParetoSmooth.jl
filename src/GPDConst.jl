for len in [8, 16, 32, 64, 128, 256, 512, 1024]
    sample = -log.(1 .- collect(1:len) ./ (len+1))
    x, y, weights = ParetoSmooth.gpdfit(sample; min_grid_pts=len)

    mass = 0.
    eval(Meta.parse(eval("global CUTPOINTS_$len = zeros(len)")))
    bins = len
    n=1
    for i in 1:length(weights)
        mass += weights[i]
        if mass ≥ 1/bins
            mass -= 1/bins
            global proportion = (i - mass) / length(weights)
            eval(Meta.parse(eval("CUTPOINTS_$len[n] = proportion")))
            n += 1
        end
    end
end