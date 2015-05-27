require 'nn'
require 'image'
require 'optim'

P = 11
number_classes = 2

opt = {batchSize = 200,
       momentum = 0,
       learningRate = 0.05}

image_length = 32
geometry = {image_length,image_length}

model = nn.Sequential()

function convolution_length(ol,f)
   return ol-f+1
end

function pooling_length(ol,p)
   return math.floor((ol-p)/p)+1
end


model:add(nn.SpatialConvolutionMM(1, 32, 5, 5))
l = convolution_length(image_length,5) -- l = dimension of the images at current layer
model:add(nn.Tanh())
model:add(nn.SpatialMaxPooling(3, 3, 3, 3))
l = pooling_length(l,3)
-- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
model:add(nn.SpatialConvolutionMM(32, 64, 5, 5))
l = convolution_length(l,5)
model:add(nn.Tanh())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
l = pooling_length(l,2)
-- stage 3 : standard 2-layer MLP:
model:add(nn.Reshape(64*l*l))
model:add(nn.Linear(64*l*l, 200))
model:add(nn.Tanh())
model:add(nn.Linear(200, 2))
model:add(nn.LogSoftMax())

parameters,gradParameters = model:getParameters()

criterion = nn.ClassNLLCriterion()
confusion = optim.ConfusionMatrix(number_classes)



-- training function
function train(dataset)
   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- do one epoch
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,dataset:size(),opt.batchSize do
      -- create mini batch
      local inputs = torch.Tensor(opt.batchSize,1,geometry[1],geometry[2])
      local targets = torch.Tensor(opt.batchSize)
      local k = 1
      for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
         -- load new sample
         local sample = dataset[i]
         local input = sample[1]:clone():reshape(1,geometry[1],geometry[2])
         local target = sample[2]
         inputs[k] = input
         targets[k] = target
         k = k + 1
      end
      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
         -- just in case:
         collectgarbage()

         -- get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end

         -- reset gradients
         gradParameters:zero()

         -- evaluate function for complete mini batch
         local outputs = model:forward(inputs)
         local f = criterion:forward(outputs, targets)

         -- estimate df/dW
         local df_do = criterion:backward(outputs, targets)
         model:backward(inputs, df_do)

         -- update confusion
         for i = 1,opt.batchSize do
            confusion:add(outputs[i], targets[i])
         end

         -- return f and df/dX
         return f,gradParameters
      end


      -- Perform SGD step:
      sgdState = sgdState or {
	 learningRate = opt.learningRate,
	 momentum = opt.momentum,
	 learningRateDecay = 5e-7
			     }
      optim.sgd(feval, parameters, sgdState)
      
      -- disp progress
      xlua.progress(t, dataset:size())

   end
   
   -- time taken
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   print(confusion.totalValid * 100)
   confusion:zero()

   -- next epoch
   epoch = epoch + 1
end

function test(dataset)
   -- local vars
   local time = sys.clock()
   -- test over given dataset
   print('<trainer> on testing Set:')
   for t = 1,dataset:size(),opt.batchSize do
      -- disp progress
      xlua.progress(t, dataset:size())
      -- create mini batch
      local inputs = torch.Tensor(opt.batchSize,1,geometry[1],geometry[2])
      local targets = torch.Tensor(opt.batchSize)
      local k = 1
      for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
	 -- load new sample
	 local sample = dataset[i]
	 local input = sample[1]:clone():reshape(1,geometry[1],geometry[2])
	 local target = sample[2]
	 inputs[k] = input
	 targets[k] = target
	 k = k + 1
      end
      -- test samples
      local preds = model:forward(inputs)
      -- confusion:
      for i = 1,opt.batchSize do
	 confusion:add(preds[i], targets[i])
      end
   end
   -- timing
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')
   -- print confusion matrix
   print(confusion)
   print(confusion.totalValid * 100)
   print('% mean class accuracy (test set)')
   confusion:zero()
end
function load_image(problem,class,index)
   local filename = string.format('../svrt/results_problem_%i/sample_%i_%04i.png', problem, class, index)
   local img_raw = image.load(filename,1)
   local i = image.scale(img_raw,geometry[1],geometry[2])
   i = -i
   i = i+1
   return i
end

-- load and preprocess images



training = {}
maximum_index = 100
function training:size() return 2*maximum_index end
for j = 1, maximum_index do 
   training[j] = {load_image(P,1,j-1),1}
   training[j+maximum_index] = {load_image(P,0,j-1),2}
end


for e = 1,100 do 
   train(training)
--   test(training)
end
