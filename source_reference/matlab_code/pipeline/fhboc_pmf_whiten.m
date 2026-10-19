function whitened = fhboc_pmf_whiten(model,values)
if isempty(model.whitening),whitened=model.lower\values(model.permutation,:);
else,whitened=model.whitening*values;end
end
