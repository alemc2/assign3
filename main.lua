--
----  Copyright (c) 2014, Facebook, Inc.
----  All rights reserved.
----
----  This source code is licensed under the Apache 2 license found in the
----  LICENSE file in the root directory of this source tree. 
----

require('nngraph')
require('base')
require('optim')
ptb = require('data')

--Get command line params
cmd = torch.CmdLine()
cmd:text()
cmd:option('-save','models/baseline.net','model save file')
cmd:option('-dropout',0,'dropout probability - 0 means no dropout')
cmd:option('-cell','lstm','lstm/gru cell')
cmd:option('-gpu',false,'use gpu')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS | ADAM | ADAGRAD | ADADELTA')
cmd:option('-learningRate', 1, 'learning rate at t=0')
cmd:option('-learningRateDecay', 0, 'learning rate decay')
cmd:option('-beta1', 0.9, 'beta1 (for Adam)')
cmd:option('-beta2', 0.999, 'beta2 (for Adam)')
cmd:option('-epsilon', 1e-8, 'epsilon (for Adam)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-nesterov', false, 'nesterov acceleration momentum (SGD only)')
cmd:option('-dampening', 0, 'momentum dampening (SGD only)')
cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
opt = cmd:parse(arg or {})

if opt.gpu then
    require 'cunn'
    print("Running on GPU") 
    
else
    require 'nn'
    print("Running on CPU")
end

savefile = string.split(opt.save,'%.')

print '==> configuring optimizer'

if opt.optimization == 'CG' then
   optimState = {
      maxIter = opt.maxIter
   }
   optimMethod = optim.cg

elseif opt.optimization == 'LBFGS' then
   optimState = {
      learningRate = opt.learningRate,
      maxIter = opt.maxIter,
      nCorrection = 10
   }
   optimMethod = optim.lbfgs

elseif opt.optimization == 'SGD' then
   optimState = {
      learningRate = opt.learningRate,
      weightDecay = opt.weightDecay,
      momentum = opt.momentum,
      learningRateDecay = opt.learningRateDecay,
      nesterov = opt.nesterov,
      dampening = opt.dampening
   }
   optimMethod = optim.sgd

elseif opt.optimization == 'ASGD' then
   optimState = {
      eta0 = opt.learningRate,
      t0 = trsize * opt.t0
   }
   optimMethod = optim.asgd

elseif opt.optimization == 'ADAM' then
   optimState = {
      learningRate = opt.learningRate,
      beta1 = opt.beta1,
      beta2 = opt.beta2,
      epsilon = opt.epsilon
   }
   optimMethod = optim.adam

elseif opt.optimization == 'ADAGRAD' then
   optimState = {
      learningRate = opt.learningRate,
   }
   optimMethod = optim.adagrad

elseif opt.optimization == 'ADADELTA' then
   optimState = {
      -- rho = ... interpolation parameter, add if needed
      -- eps = ... for numerical stability, add if needed
   }
   optimMethod = optim.adadelta

else
   error('unknown optimization method')
end

-- Trains 1 epoch and gives validation set ~182 perplexity (CPU).
local params = {
                batch_size=20, -- minibatch
                seq_length=20, -- unroll length
                layers=2,
                decay=2,
                rnn_size=200, -- hidden unit size
                dropout=opt.Dropout,
                init_weight=0.1, -- random weight initialization limits
                lr=1, --learning rate
                vocab_size=10000, -- limit on the vocabulary size
                max_epoch=4,  -- when to start decaying learning rate
                max_max_epoch=13, -- final epoch
                max_grad_norm=5 -- clip when gradients exceed this norm value
               }

if opt.cell == 'lstm' then
    params.num_s = 2*params.layers
elseif opt.cell == 'gru' then
    params.num_s = params.layers
end

function transfer_data(x)
    if opt.gpu then
        return x:cuda()
    else
        return x
    end
end

model = {}

local function lstm(x, prev_c, prev_h)
    -- Calculate all four gates in one go
    local i2h              = nn.Linear(params.rnn_size, 4*params.rnn_size)(x)
    local h2h              = nn.Linear(params.rnn_size, 4*params.rnn_size)(prev_h)
    local gates            = nn.CAddTable()({i2h, h2h})

    -- Reshape to (batch_size, n_gates, hid_size)
    -- Then slize the n_gates dimension, i.e dimension 2
    local reshaped_gates   =  nn.Reshape(4,params.rnn_size)(gates)
    local sliced_gates     = nn.SplitTable(2)(reshaped_gates)

    -- Use select gate to fetch each gate and apply nonlinearity
    local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
    local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
    local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
    local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
    })
    local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    return next_c, next_h
end

local function gru(x, prev_h)
    -- Calculate all two gates in one go. 3rd is for the transform
    local i2h              = nn.Linear(params.rnn_size, 3*params.rnn_size)(x)
    local h2h              = nn.Linear(params.rnn_size, 3*params.rnn_size)(prev_h)
    -- For gates use only the first 2 parts, don't need to sum like this for the 3rd part as it is based on reset gate
    local gates            = nn.CAddTable()({
                                            nn.Narrow(2,1,2*params.rnn_size)(i2h),
                                            nn.Narrow(2,1,2*params.rnn_size)(h2h)
                                        })

    -- Reshape to (batch_size, n_gates, hid_size)
    -- Then slice the n_gates dimension, i.e dimension 2
    local reshaped_gates   =  nn.Reshape(2, params.rnn_size)(gates)
    local sliced_gates     = nn.SplitTable(2)(reshaped_gates)

    -- Use select gate to fetch each gate and apply nonlinearity
    -- reset gate r
    local reset_gate       = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
    -- update gate z
    local update_gate      = nn.Sigmoid()(nn.SelectTable(2)(sliced_gates))
    
    local reset_h          = nn.CMulTable()({reset_gate, nn.Narrow(2, 2*params.rnn_size+1, params.rnn_size)(h2h)})
    local transform_inp    = nn.CAddTable()({nn.Narrow(2, 2*params.rnn_size+1, params.rnn_size)(i2h), reset_h})
    local h_out            = nn.Tanh()(transform_inp)
    

    -- Use of formula next_h = (1-z)*prev_h + z*h_out
    local h_feedback       = nn.CMulTable()({update_gate, nn.CSubTable()({h_out, prev_h})})
    local next_h           = nn.CAddTable()({prev_h, h_feedback})

    return next_h
end

function create_network()
    local x                  = nn.Identity()()
    local y                  = nn.Identity()()
    local prev_s             = nn.Identity()()
    local i                  = {[0] = nn.LookupTable(params.vocab_size,
                                                    params.rnn_size)(x)}
    local next_s             = {}
    local split              = {prev_s:split(params.num_s)}
    for layer_idx = 1, params.layers do
        local dropped        = nn.Dropout(params.dropout)(i[layer_idx - 1])
        if opt.cell == 'lstm' then
            local prev_c         = split[2 * layer_idx - 1]
            local prev_h         = split[2 * layer_idx]
            local next_c, next_h = lstm(dropped, prev_c, prev_h)
            table.insert(next_s, next_c)
            table.insert(next_s, next_h)
            i[layer_idx] = next_h
        elseif opt.cell == 'gru' then
            local prev_h         = split[layer_idx]
            local next_h = gru(dropped, prev_h)
            table.insert(next_s, next_h)
            i[layer_idx] = next_h
        end
    end
    local h2y                = nn.Linear(params.rnn_size, params.vocab_size)
    local dropped            = nn.Dropout(params.dropout)(i[params.layers])
    local h2y_dropped        = h2y(dropped)
    local pred_prob          = nn.SoftMax()(h2y_dropped)
    --local pred               = nn.Log()(nn.Clamp(1e-15,1)(pred_prob))
    local pred               = nn.LogSoftMax()(h2y_dropped)
    local err                = nn.ClassNLLCriterion()({pred, y})
    local module             = nn.gModule({x, y, prev_s},
                                      {err, nn.Identity()(next_s),pred_prob})
    -- initialize weights
    module:getParameters():uniform(-params.init_weight, params.init_weight)
    return transfer_data(module)
end

function setup()
    print("Creating a RNN "..opt.cell.." network.")
    local core_network = create_network()
    paramx, paramdx = core_network:getParameters()
    model.s = {}
    model.ds = {}
    model.start_s = {}
    for j = 0, params.seq_length do
        model.s[j] = {}
        for d = 1, params.num_s do
            model.s[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
        end
    end
    for d = 1, params.num_s do
        model.start_s[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
        model.ds[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    end
    model.core_network = core_network
    model.rnns = g_cloneManyTimes(core_network, params.seq_length)
    model.norm_dw = 0
    model.err = transfer_data(torch.zeros(params.seq_length))
end

function reset_state(state)
    state.pos = 1
    if model ~= nil and model.start_s ~= nil then
        for d = 1, params.num_s do
            model.start_s[d]:zero()
        end
    end
end

function reset_ds()
    for d = 1, #model.ds do
        model.ds[d]:zero()
    end
end

function fp(state)
    -- g_replace_table(from, to).  
    g_replace_table(model.s[0], model.start_s)
    
    -- reset state when we are done with one full epoch
    if state.pos + params.seq_length > state.data:size(1) then
        reset_state(state)
    end
    
    -- forward prop
    for i = 1, params.seq_length do
        local x = state.data[state.pos]
        local y = state.data[state.pos + 1]
        local s = model.s[i - 1]
        model.err[i], model.s[i], _ = unpack(model.rnns[i]:forward({x, y, s}))
        state.pos = state.pos + 1
    end
    
    -- next-forward-prop start state is current-forward-prop's last state
    g_replace_table(model.start_s, model.s[params.seq_length])
    
    -- cross entropy error
    return model.err:mean()
end

function bp(state)
    -- start on a clean slate. Backprop over time for params.seq_length.
    paramdx:zero()
    reset_ds()
    for i = params.seq_length, 1, -1 do
        -- to make the following code look almost like fp
        state.pos = state.pos - 1
        local x = state.data[state.pos]
        local y = state.data[state.pos + 1]
        local s = model.s[i - 1]
        -- Why 1? - Because nngraph converts criteria to a module and hence it's second argument is gradout not target which for the case of the end criteria is 1.
        local derr = transfer_data(torch.ones(1))
        -- The predicted probability is only to extract the output, it in no way affects the loss and hence shouldn't affect the weights so gradout is 0
        local dpred = transfer_data(torch.zeros(params.batch_size, params.vocab_size))
        -- tmp stores the ds
        local tmp = model.rnns[i]:backward({x, y, s},
                                           {derr, model.ds, dpred})[3]
        -- remember (to, from)
        g_replace_table(model.ds, tmp)
    end
    
    -- undo changes due to changing position in bp
    state.pos = state.pos + params.seq_length
    
    -- gradient clipping
    model.norm_dw = paramdx:norm()
    if model.norm_dw > params.max_grad_norm then
        local shrink_factor = params.max_grad_norm / model.norm_dw
        paramdx:mul(shrink_factor)
    end
    
    -- gradient descent step
    local feval = function(x)
        return model.rnns[params.seq_length].output,paramdx
    end
    if optimMethod == optim.asgd then
        _,_,average = optimMethod(feval, paramx, optimState)
    else
        optimMethod(feval, paramx, optimState)
    end
    --paramx:add(paramdx:mul(-params.lr))
end

function run_valid()
    -- again start with a clean slate
    reset_state(state_valid)
    
    -- no dropout in testing/validating
    g_disable_dropout(model.rnns)
    
    -- collect perplexity over the whole validation set
    local len = (state_valid.data:size(1) - 1) / (params.seq_length)
    local perp = 0
    for i = 1, len do
        perp = perp + fp(state_valid)
    end
    print("Validation set perplexity : " .. g_f3(torch.exp(perp / len)))
    g_enable_dropout(model.rnns)
end

function run_test()
    reset_state(state_test)
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
    end
    print("Test set perplexity : " .. g_f3(torch.exp(perp / (len - 1))))
    g_enable_dropout(model.rnns)
end

if opt.gpu then
    g_init_gpu({})
end

-- get data in batches
state_train = {data=transfer_data(ptb.traindataset(params.batch_size))}
state_valid =  {data=transfer_data(ptb.validdataset(params.batch_size))}
state_test =  {data=transfer_data(ptb.testdataset(params.batch_size))}

print("Network parameters:")
print(params)

local states = {state_train, state_valid, state_test}
for _, state in pairs(states) do
    reset_state(state)
end
setup()
step = 0
epoch = 0
total_cases = 0
beginning_time = torch.tic()
start_time = torch.tic()
print("Starting training.")
words_per_step = params.seq_length * params.batch_size
epoch_size = torch.floor(state_train.data:size(1) / params.seq_length)

while epoch < params.max_max_epoch do

    -- take one step forward
    perp = fp(state_train)
    if perps == nil then
        perps = torch.zeros(epoch_size):add(perp)
    end
    perps[step % epoch_size + 1] = perp
    step = step + 1
    
    -- gradient over the step
    bp(state_train)
    
    -- words_per_step covered in one step
    total_cases = total_cases + params.seq_length * params.batch_size
    epoch = step / epoch_size
    
    -- display details at some interval
    if step % torch.round(epoch_size / 10) == 10 then
        wps = torch.floor(total_cases / torch.toc(start_time))
        since_beginning = g_d(torch.toc(beginning_time) / 60)
        print('epoch = ' .. g_f3(epoch) ..
             ', train perp. = ' .. g_f3(torch.exp(perps:mean())) ..
             ', wps = ' .. wps ..
             ', dw:norm() = ' .. g_f3(model.norm_dw) ..
             ', lr = ' ..  g_f3(optimState.learningRate) ..
             ', since beginning = ' .. since_beginning .. ' mins.')
    end
    
    -- run when epoch done
    if step % epoch_size == 0 then
        run_valid()
        if epoch > params.max_epoch and opt.optimization == 'SGD'  then
            optimState.learningRate = optimState.learningRate / params.decay
            --params.lr = params.lr / params.decay
        end
        -- Save models at this stage so that we can premature exit if needed
        torch.save(string.format('%s_%d.%s',savefile[1],epoch,savefile[2]),model)
    end
end
run_test()
print("Training is over.")
