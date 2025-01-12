--
----  Copyright (c) 2014, Facebook, Inc.
----  All rights reserved.
----
----  This source code is licensed under the Apache 2 license found in the
----  LICENSE file in the root directory of this source tree. 
----

local stringx = require('pl.stringx')
local file = require('pl.file')

local ptb_path = "./data/"

local trainfn = ptb_path .. "ptb.train.txt"
local testfn  = ptb_path .. "ptb.test.txt"
local validfn = ptb_path .. "ptb.valid.txt"

local vocab_idx = 0
local vocab_map = {}
local vocab_inverse_map = {}

-- Stacks replicated, shifted versions of x_inp
-- into a single matrix of size x_inp:size(1) x batch_size.
local function replicate(x_inp, batch_size)
    local s = x_inp:size(1)
    local x = torch.zeros(torch.floor(s / batch_size), batch_size)
    for i = 1, batch_size do
        local start = torch.round((i - 1) * s / batch_size) + 1
        local finish = start + x:size(1) - 1
        x:sub(1, x:size(1), i, i):copy(x_inp:sub(start, finish))
    end
    return x
end

local function load_data(fname)
    local data = file.read(fname)
    data = stringx.replace(data, '\n', '<eos>')
    data = stringx.split(data)
    --print(string.format("Loading %s, size of data = %d", fname, #data))
    local x = torch.zeros(#data)
    for i = 1, #data do
        if vocab_map[data[i]] == nil then
            vocab_idx = vocab_idx + 1
            vocab_map[data[i]] = vocab_idx
            vocab_inverse_map[vocab_idx] = data[i]
        end
        x[i] = vocab_map[data[i]]
    end
    return x
end

local function traindataset(batch_size, char)
   local x = load_data(trainfn)
   x = replicate(x, batch_size)
   return x
end

-- Intentionally we repeat dimensions without offseting.
-- Pass over this batch corresponds to the fully sequential processing.
local function testdataset(batch_size)
    if testfn then
        local x = load_data(testfn)
        x = x:resize(x:size(1), 1):expand(x:size(1), batch_size)
        return x
    end
end

local function validdataset(batch_size)
    local x = load_data(validfn)
    x = replicate(x, batch_size)
    return x
end

-- Assume that vocab is already populated. If word not in dictionary then its <unk>
local function encode_data(data,batch_size) --gets table returns tensor
    local x = torch.zeros(#data)
    for i = 1, #data do
        if vocab_map[data[i]] == nil then
            data[i] = '<unk>'
        end
        x[i] = vocab_map[data[i]]
    end
    x = x:resize(x:size(1), 1):expand(x:size(1), batch_size)
    return x
end

local function decode_data(data) --gets tensor return table
    local x = {}
    for i = 1, data:size(1) do
        x[i] = vocab_inverse_map[data[i]]
    end
    return x
end

return {traindataset=traindataset,
        testdataset=testdataset,
        validdataset=validdataset,
        encode_data=encode_data,
        decode_data=decode_data,
        vocab_map=vocab_map,
        vocab_inverse_map=vocab_inverse_map}
