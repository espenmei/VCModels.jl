
function savemodel(m::VCModel, filename::String)
    file = jldopen(filename, "w")
    write(file, "model", m)
    close(file)
end

function loadmodel(filename)
    inp = JLD.load(filename)
    inp["model"]
end