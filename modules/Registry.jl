module Registry

export PatientRecord, build_registry

struct PatientRecord
    patient::String
    session::String
    label::Symbol # :healthy, :infarction, :unknown
    age::Union{Int, Missing}
    sex::Union{String, Missing}
    acute_infarction::Union{String, Missing}
    former_infarction::Union{String, Missing}
    additional_dx::Union{String, Missing}
end

function parse_hea(path::String)::Union{PatientRecord, Nothing}
    lines = readlines(path)
    isempty(lines) && return nothing

    session = split(lines[1])[1]
    patient = replace(basename(dirname(path)), "patient" => "")
    session = replace(basename(path), ".hea" => "")

    age = missing
    sex = missing
    acute_inf = missing
    former_inf = missing
    additional_dx = missing
    reason = missing

    for line in lines
        line = strip(line)
        startswith(line, "#") || continue
        content = strip(line[2:end])

        if startswith(content, "age:")
            val = strip(join(split(content, ":")[2:end], ":"))
            parsed = tryparse(Int, val)
            age = parsed === nothing ? missing : parsed

        elseif startswith(content, "sex:")
            val = strip(join(split(content, ":")[2:end], ":"))
            sex = isempty(val) ? missing : val

        elseif startswith(content, "Reason for admission:")
            val = strip(join(split(content, ":")[2:end], ":"))
            reason = isempty(val) ? missing : lowercase(val)

        elseif startswith(content, "Acute infarction (localization):")
            val = strip(join(split(content, ":")[2:end], ":"))
            acute_inf = isempty(val) ? missing : lowercase(val)

        elseif startswith(content, "Former infarction (localization):")
            val = strip(join(split(content, ":")[2:end], ":"))
            former_inf = isempty(val) ? missing : lowercase(val)

        elseif startswith(content, "Additional diagnoses:")
            val = strip(join(split(content, ":")[2:end], ":"))
            additional_dx = isempty(val) ? missing : lowercase(val)
        end
    end

    label = classify_label(reason, acute_inf, former_inf)
    return PatientRecord(patient, session, label, age, sex, acute_inf, former_inf, additional_dx)
end

function classify_label(reason, acute_inf, former_inf)::Symbol
    # Healthy
    if !ismissing(reason) && contains(reason, "healthy")
        return :healthy
    end

    # Infarction
    has_acute = !ismissing(acute_inf)  && acute_inf  ∉ ("no", "n/a", "")
    has_former = !ismissing(former_inf) && former_inf ∉ ("no", "n/a", "")
    if has_acute || has_former
        return :infarction
    end

    return :unknown
end

function build_registry(db_root::String)::Vector{PatientRecord}
    records = PatientRecord[]

    for patient_dir in readdir(db_root; join=true)
        isdir(patient_dir) || continue
        for file in readdir(patient_dir; join=true)
            endswith(file, ".hea") || continue
            rec = parse_hea(file)
            rec === nothing && continue
            push!(records, rec)
        end
    end

    sort!(records, by = r -> r.patient)
    return records
end

end # module Registry