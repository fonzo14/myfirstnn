NN_ROOT = File.join(File.dirname(__FILE__), '..')

Dir.glob("#{File.join(NN_ROOT, "lib", "java","**")}/*.jar").each { |jar| require jar }

java_import 'org.ejml.simple.SimpleMatrix'
java_import 'java.util.Random'

require "multi_json"

require_relative 'myfirstnn/utils'
require_relative 'myfirstnn/tokenizer'
require_relative 'myfirstnn/vocabulary'
require_relative 'myfirstnn/embeddings'

require_relative 'myfirstnn/model_serializer'
require_relative 'myfirstnn/backpropagation_task'
require_relative 'myfirstnn/sample'
require_relative 'myfirstnn/samples_loader'
require_relative 'myfirstnn/neural_network'
