all_params = { ...
    { ...
        [40, 30], ...
        [[10, 10]; [20, 15]; [15, 17]; [37, 2]; [1, 27]], ...
        [20, 21, inf, 9, 5], ...
        [39, 3, 21, 13, 1], ...
        [[1, 30]; [37, 12]; [24, 24]; [5, 17]; [8, 8]], ...
    } ...
}; 

for idx=1:length(all_params)
    disp(idx);
    params = all_params{idx};
    
    multi_arc = makeMultiArc(params{:});
    
    idx_padded = sprintf('%06d', idx - 1);
    filename = ['collectedValues/makeMultiArc/' idx_padded '.mat'];
    save(filename, 'params', 'multi_arc');
end
