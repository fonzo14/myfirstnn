NN_ROOT = File.join(File.dirname(__FILE__), '..')

Dir.glob("#{File.join(NN_ROOT, "lib", "java","**")}/*.jar").each { |jar| require jar }

%w{neural_network embeddings tokenizer samples_loader}.each do |f|
  require "mydl4j/#{f}"
end
