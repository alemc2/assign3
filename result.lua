stringx = require('pl.stringx')
ptb = require('data')
require('nngraph')
require('base')

--Get command line params
cmd = torch.CmdLine()
cmd:text()
cmd:option('-model','models/baseline_best.net','model file')
cmd:option('-gpu',false,'use gpu')
opt = cmd:parse(arg or {})

if opt.gpu then
    require 'cunn'
    print("Running on GPU") 
    
else
    require 'nn'
    print("Running on CPU")
end

local params = {
                batch_size=20, -- minibatch
                seq_length=20, -- unroll length
                layers=2,
                rnn_size=200, -- hidden unit size
                vocab_size=10000, -- limit on the vocabulary size
               }

function transfer_data(x)
    if gpu then
        return x:cuda()
    else
        return x
    end
end

function run_test()
    state_test.pos = 1
    g_disable_dropout(model.rnns)
    local perp = 0
    local len = state_test.data:size(1)
    
    -- no batching here
    g_replace_table(model.s[0], model.start_s)
    for i = 1, (len - 1) do
        local x = state_test.data[i]
        local y = state_test.data[i + 1]
        perp_tmp, model.s[1] = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
        perp = perp + perp_tmp[1]
        g_replace_table(model.s[0], model.s[1])
        print(i)
    end
    print("Test set perplexity : " .. g_f3(torch.exp(perp / (len - 1))))
    g_enable_dropout(model.rnns)
end

if opt.gpu then
    g_init_gpu({})
end

-- get data in batches, essentially load up the data to fill the vocabulary
state_train = {data=transfer_data(ptb.traindataset(params.batch_size))}
state_valid =  {data=transfer_data(ptb.validdataset(params.batch_size))}
state_test =  {data=transfer_data(ptb.testdataset(params.batch_size))}

model = torch.load(opt.model)
run_test()
