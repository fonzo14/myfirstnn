java_import 'java.util.concurrent.Executors'

class NeuralNetwork
  attr_reader :vocabulary, :hidden_layer_size, :output_layer_size
  attr_reader :w_hidden, :b_hidden, :w_output, :b_output

  def initialize(vocabulary_size, hidden_size, output_size)
    # input layer size = vocabulary size
    @vocabulary_size   = vocabulary_size
    @hidden_layer_size = hidden_size
    @output_layer_size = output_size

    @vocabulary     = Vocabulary.new(1000, 5, File.join(NN_ROOT, 'data', 'vocabulary.txt'))
    @samples_loader = SamplesLoader.new(@vocabulary, Tokenizer.new)

    # init the weight and biais
    @w_hidden = SimpleMatrix.random(@hidden_layer_size, @vocabulary.size, -1.0, 1.0, Random.new)
    @b_hidden = SimpleMatrix.new(@hidden_layer_size, 1)

    @w_output = SimpleMatrix.random(@output_layer_size, @hidden_layer_size, -1.0, 1.0, Random.new)
    @b_output = SimpleMatrix.new(@output_layer_size, 1)
  end

  def train(epochs, batch_size, learning_rate)
    @executor = Executors.newFixedThreadPool(8)

    # load train data
    samples = @samples_loader.load("train/train.txt")

    # evaluate before training
    # evaluate(@samples_loader.load("train/evaluation.txt"))

    epochs.times do |epoch|
      start_epoch = Time.now.to_i
      puts "train epoch #{epoch}"
      samples.each_slice(batch_size).each_with_index do |batch_samples, batch_index|
        train_batch(batch_index, batch_samples, learning_rate)
        if ((batch_index + 1) % 50) == 0
          puts "batch learning rate:#{learning_rate} epoch:#{epoch} batch_index:#{batch_index}"
          evaluate(@samples_loader.load("train/evaluation.txt"))
        end
      end

      end_epoch = Time.now.to_i

      puts "epoch #{epoch} trained in #{end_epoch - start_epoch}s"

      # evaluate at the end of epoch
      evaluate(@samples_loader.load("train/evaluation.txt"))
    end
  end

  def test
    evaluate(@samples_loader.load("train/test.txt"))
  end

  def shutdown
    @executor.shutdown
    ser = ModelSerializer.new
    ser.save(self)
  end

  private
  def train_batch(batch_index, samples, learning_rate)
    batch_size = samples.size

    # init batch delta's
    batch_delta_w_output = SimpleMatrix.new(@output_layer_size, @hidden_layer_size)
    batch_delta_b_output = SimpleMatrix.new(@output_layer_size, 1)
    batch_delta_w_hidden = SimpleMatrix.new(@hidden_layer_size, @vocabulary.size)
    batch_delta_b_hidden = SimpleMatrix.new(@hidden_layer_size, 1)

    futures = samples.map do |sample|
      task = BackpropagationTask.new(@w_hidden, @b_hidden, @w_output, @b_output, sample)
      @executor.submit(task)
    end

    futures.map { |f| f.get }.each do |backpropagation|
      delta_w_output, delta_b_output, delta_w_hidden, delta_b_hidden = backpropagation

      batch_delta_w_output = batch_delta_w_output.plus(delta_w_output)
      batch_delta_b_output = batch_delta_b_output.plus(delta_b_output)
      batch_delta_w_hidden = batch_delta_w_hidden.plus(delta_w_hidden)
      batch_delta_b_hidden = batch_delta_b_hidden.plus(delta_b_hidden)
    end

    # update weights and biais
    @w_output = @w_output.minus(Utils::new_matrix(batch_delta_w_output) { |v| (learning_rate / batch_size) * v })
    @b_output = @b_output.minus(Utils::new_matrix(batch_delta_b_output) { |v| (learning_rate / batch_size) * v })
    @w_hidden = @w_hidden.minus(Utils::new_matrix(batch_delta_w_hidden) { |v| (learning_rate / batch_size) * v })
    @b_hidden = @b_hidden.minus(Utils::new_matrix(batch_delta_b_hidden) { |v| (learning_rate / batch_size) * v })
  end

  def backpropagation(batch_index, sample)
    # backpropagation implementation

    # feedforwarding sample
    # keep the intermediate value's computation :
    # z = weighted input layer l
    # a = activation layer l
    # so that we can compute the error/gradients
    hidden_z = @w_hidden.mult(sample.data).plus(@b_hidden)
    hidden_a = Utils::tanh(hidden_z)

    output_z = @w_output.mult(hidden_a).plus(@b_output)
    output_a = Utils::softmax(output_z)

    # compute the output error / delta
    # delta = loss_derivative(output_a, sample.label).elementMult(softmax_derivative(output_z))
    # looks like derivative of the cross-entropy cost function for the softmax function is just :
    delta_output = output_a.minus(sample.label)

    # output biais delta = delta
    delta_b_output = delta_output

    # output weight delta
    delta_w_output = delta_output.mult(hidden_a.transpose)

    # hidden layer
    delta_hidden   = @w_output.transpose.mult(delta_output).elementMult(Utils::tanh_derivative(hidden_z))

    delta_b_hidden = delta_hidden
    delta_w_hidden = delta_hidden.mult(sample.data.transpose)

    [delta_w_output, delta_b_output, delta_w_hidden, delta_b_hidden]
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
    max_klass = [0,1].max_by { |k| klass.get(k,0).to_f }
    max_label = [0,1].max_by { |k| label.get(k,0).to_f }
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
    hidden_z = @w_hidden.mult(input).plus(@b_hidden)
    hidden_output = Utils::tanh(hidden_z)
    output_output = Utils::softmax(@w_output.mult(hidden_output).plus(@b_output))
    output_output
  end
end
