class ModelSerializer
  def initialize
  end

  def save(nn)
    model = {
      layers: []
    }

    [
      { size: nn.vocabulary_size, name: "input" },
      { size: nn.hidden_layer_size, name: "hidden", weights: nn.w_hidden, biases: nn.b_hidden },
      { size: nn.output_layer_size, name: "output", weights: nn.w_output, biases: nn.b_output }
    ].each do |layer|
      if layer.key?(:weights)
        [:weights, :biases].each do |m|
          elements = []
          layer[m].numRows.times do |i|
            layer[m].numCols.times do |j|
              elements <<([i, j , layer[m].get(i,j)])
            end
          end
          layer[m] = elements
        end
      end
      model[:layers] << layer
    end

    file = File.join(::NN_ROOT, 'data', 'models', "#{Time.now.to_i}.json")
    File.open(file, "w") { |f| f << MultiJson.encode(model) }
  end

  def load(name)
    file  = File.join(::NN_ROOT, 'data', 'models', "#{name}.json")

    m  = MultiJson.decode(IO.read(file))
    nn = NeuralNetwork.new(m['layers'][0]['size'], m['layers'][1]['size'], m['layers'][2]['size'])

    m['layers'][1]['weights'].each do |(row, col, value)|
      nn.w_hidden.set(row, col, value)
    end
    m['layers'][1]['biases'].each do |(row, col, value)|
      nn.b_hidden.set(row, col, value)
    end
    m['layers'][2]['weights'].each do |(row, col, value)|
      nn.w_output.set(row, col, value)
    end
    m['layers'][2]['biases'].each do |(row, col, value)|
      nn.b_output.set(row, col, value)
    end

    nn
  end
end
