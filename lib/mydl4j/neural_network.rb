java_import 'org.deeplearning4j.nn.conf.NeuralNetConfiguration'
java_import 'org.deeplearning4j.nn.multilayer.MultiLayerNetwork'

java_import 'org.deeplearning4j.nn.api.OptimizationAlgorithm'
java_import 'org.deeplearning4j.nn.weights.WeightInit'
java_import 'org.deeplearning4j.nn.conf.Updater'
java_import 'org.nd4j.linalg.lossfunctions.LossFunctions'

java_import 'org.deeplearning4j.nn.conf.layers.DenseLayer'
java_import 'org.deeplearning4j.nn.conf.layers.OutputLayer'

class NeuralNetwork
  attr_reader :input_layer_size, :hidden_layer_size, :output_layer_size

  def initialize
    input_layer_size  = 300
    hidden_layer_size = 100
    output_layer_size = 2

    conf = NeuralNetConfiguration::Builder.new
      .seed(123)
      .iterations(1)
      .optimizationAlgo(OptimizationAlgorithm::STOCHASTIC_GRADIENT_DESCENT)
      .learningRate(0.02)
      .regularization(false)
      .updater(Updater::SGD)
      .list()
      .layer(0, DenseLayer::Builder.new.nIn(input_layer_size).nOut(hidden_layer_size).weightInit(WeightInit::XAVIER).activation("relu").build())
      .layer(1, OutputLayer::Builder.new(LossFunctions::LossFunction::NEGATIVELOGLIKELIHOOD).weightInit(WeightInit::XAVIER).activation("softmax").weightInit(WeightInit::XAVIER).nIn(hidden_layer_size).nOut(output_layer_size).build())
      .pretrain(false)
      .backprop(true)
      .build()

    @model = MultiLayerNetwork.new(conf)
    @model.init

    embeddings        = Embeddings.new(File.join(NN_ROOT, 'data', 'embeddings', 'word2vec-d300-mc5-w5.model.txt'))
    @input_layer_size = embeddings.dimensions_count

    @samples_loader = SamplesLoader.new(embeddings, Tokenizer.new)

    puts "neural network initialized"
  end

  def train
    train_samples = @samples_loader.load("train/train.txt")
    eval_samples  = @samples_loader.load("train/evaluation.txt")

    25.times do |epoch|
      puts "starting epoch #{epoch}"
      train_samples.each_with_index do |dataset,i|
        @model.fit(dataset)
        #if ((i+1) % 2500).zero?
        if true
          evaluate(eval_samples)
          sleep(3)
        end
      end
    end
  end

  def test
    evaluate(@samples_loader.load("train/test.txt"))
    puts "testing done"
  end

  private
    def evaluate(samples)
      total_ok = 0
      samples.each do |dataset|
        if ok?(@model.output(dataset.getFeatures, false), dataset.getLabels)
          total_ok += 1
        end
      end
      accuracy = total_ok.to_f / samples.size
      puts "accuracy: #{accuracy}"
    end

    def ok?(klass, label)
      max_klass = [0,1].max_by { |k| klass.getFloat(0,k).to_f }
      max_label = [0,1].max_by { |k| label.getFloat(0,k).to_f }
      (max_klass == max_label)
    end
end
