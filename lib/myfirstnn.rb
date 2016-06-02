NN_ROOT = File.join(File.dirname(__FILE__), '..')

java_import 'org.ejml.simple.SimpleMatrix'
java_import 'java.util.Random'

require_relative 'myfirstnn/utils'
require_relative 'myfirstnn/tokenizer'
require_relative 'myfirstnn/vocabulary'

require_relative 'myfirstnn/sample'
require_relative 'myfirstnn/samples_loader'
require_relative 'myfirstnn/neural_network'
