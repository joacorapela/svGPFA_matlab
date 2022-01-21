function [C_Mean, C_Group, C_Noise] = extract_SubspaceMapping(m);

if isstruct(m.prs.C)
    C_Mean = m.prs.C.Mean;
    if isfield(m.prs.C,'Group')
        C_Group = m.prs.C.Group;
    else
        C_Group = [];
    end
    if isfield(m.prs.C,'Noise')
        C_Noise = m.prs.C.Noise;
    else
        C_Noise = [];
    end
else
    C_Mean = m.prs.C;
    C_Group = m.prs.C;
    C_Noise = m.prs.C;
end
