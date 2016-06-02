class NeuralNetwork
  def initialize
    @vocabulary = Vocabulary.new(1000, 5, File.join(NN_ROOT, 'data', 'vocabulary.txt'))
    @tokenizer  = Tokenizer.new

    @samples_loader = SamplesLoader.new(@vocabulary, @tokenizer)
  end

  def train(epochs, batch_size, learning_rate)
    # init the weight and biais
    # input layer size = vocabulary size
    # hidden layer = 50
    # output layer = 2

    @hidden_layer_size = 50
    @output_layer_size = 2

    @w_hidden = SimpleMatrix.random(@hidden_layer_size, @vocabulary.size, -1.0, 1.0, Random.new)
    @b_hidden = SimpleMatrix.new(@hidden_layer_size, 1)

    @w_output = SimpleMatrix.random(@output_layer_size, @hidden_layer_size, -1.0, 1.0, Random.new)
    @b_output = SimpleMatrix.new(@output_layer_size, 1)

    samples = @samples_loader.load("train/train.txt")
    epochs.times do |epoch|
      puts "train epoch #{epoch}"
      samples.shuffle.each_slice(batch_size) do |batch_samples|
        train_batch(batch_samples)
      end

      # evaluate at the end of epoch
      evaluate(@samples_loader.load("train/evaluation.txt"))
    end
  end

  def test
    evaluate(@samples_loader.load("train/test.txt"))
  end

  private
  def train_batch(samples)

  end

  def evaluate(samples)
    total_ok = 0
    samples.shuffle.each do |sample|
      if ok?(feedforward(sample.data), sample.label)
        total_ok += 1
      end
    end
    accuracy = total_ok.to_f / samples.size
    puts "accuracy: #{accuracy}"
  end

  def ok?(klass, label)
    max_klass = [0,1].max_by { |k| klass.get(k,0) }
    max_label = [0,1].max_by { |k| label.get(0,k) }
    (max_klass == max_label)
  end

  def loss(samples)
    total_loss = samples.inject(0) { |sum,sample| sum += sample_loss(sample) }
    avg_loss   = total_loss / samples.size
    avg_loss
  end

  def sample_loss(sample)
    # cross entropy / log-likehood
    prob = feedforward(sample.data)
    e1 = sample.label.mult(Utils::new_matrix(prob) { |v| Math::log(v) })
    e2 = Utils::new_matrix(sample.label) { |v| 1.0 - v }.mult(Utils::new_matrix(prob) { |v| Math::log(1.0 - v) })
    l = -e1.plus(e2).get(0,0)
    l
  end

  def feedforward(input)
    hidden_output = Utils::tanh(@w_hidden.mult(input).plus(@b_hidden))
    output_output = Utils::softmax(@w_output.mult(hidden_output).plus(@b_output))
    output_output
  end
end
