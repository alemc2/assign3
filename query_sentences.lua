stringx = require('pl.stringx')
ptb = require('data')
require 'io'
require('nngraph')
require('base')

--Get command line params
cmd = torch.CmdLine()
cmd:text()
cmd:option('-model','models/baseline_best.net','model file')
opt = cmd:parse(arg or {})

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

function readline()
  local line = io.read("*line")
  if line == nil then error({code="EOF"}) end
  line = stringx.split(line)
  local number = table.remove(line,1)
  if tonumber(number) == nil then error({code="init"}) end
  line = ptb.encode_data(line,params.batch_size)
  return {data=line,number=number}
end

function predict_text(inp_data,number)
    g_disable_dropout(model.rnns)
    local perp = 0
    local data_size = inp_data:size()
    local len = data_size[1]
    data_size[1] = data_size[1] + number
    local data = torch.zeros(data_size)
    data[{{1,len},{}}] = inp_data
    -- no batching here
    g_replace_table(model.s[0], model.start_s)
    for i = 1, (len - 1) do
        local x = data[i]
        local y = data[i + 1]
        perp_tmp, model.s[1], _ = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
        perp = perp + perp_tmp[1]
        g_replace_table(model.s[0], model.s[1])
    end
    print("Input string perplexity : " .. g_f3(torch.exp(perp / (len - 1))))
    for i = len,len+number-1 do
        local x = data[i]
        local y = data[i] -- Just dummy, we won't use perplexity here
        local pred_prob
        _, model.s[1], pred_prob = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
        local sampled_idx = torch.multinomial(pred_prob[1],1)[1]    --Use pred_prob[1] as it is batch size big with repeats so we just want to sample once.
        data[i+1] = torch.ones(1,params.batch_size):mul(sampled_idx)
    end
    g_enable_dropout(model.rnns)
    return data[{{},1}] --return 1st column, everything else is repeat anyways
end


-- get data in batches, essentially load up the data to fill the vocabulary
state_train = {data=transfer_data(ptb.traindataset(params.batch_size))}
state_valid =  {data=transfer_data(ptb.validdataset(params.batch_size))}
state_test =  {data=transfer_data(ptb.testdataset(params.batch_size))}

model = torch.load(opt.model)

while true do
  print("Query: len word1 word2 etc")
  local ok, line = pcall(readline)
  if not ok then
    if line.code == "EOF" then
      break -- end loop
    elseif line.code == "vocab" then
      print("Word not in vocabulary", line.word)
    elseif line.code == "init" then
      print("Start with a number")
    else
      print(line)
      print("Failed, try again")
    end
  else
    local text_encoded = predict_text(line.data,line.number)
    local text = ptb.decode_data(text_encoded)
    local printstr = stringx.join(' ',text)
    io.write(printstr)
    io.write('\n')
  end
end
