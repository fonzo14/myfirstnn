java_import 'org.nd4j.linalg.dataset.DataSet'

class SamplesLoader
  def initialize(embeddings, tokenizer)
    @embeddings = embeddings
    @tokenizer  = tokenizer
  end

  def load(file)
    datasets = []

    IO.foreach(File.join(NN_ROOT, 'data', file)) do |line|
      text, category = line.chomp.split("\t")

      h = Hash.new(0)
      tokens = @tokenizer.tokenize(text).select { |word| @embeddings.include?(word) }
      tokens.each do |word|
        @embeddings.values(word).each_with_index do |v,i|
          h[i] += v
        end
      end

      if h.any?
        values = []

        h.each do |k,v|
          z = v / tokens.size
          values << z
        end

        data = Java::OrgNd4jLinalgFactory::Nd4j.create(values.to_java(Java::float), [1,300].to_java(Java::int))

        label_values = []
        if category == 'SPORT'
          label_values = [1.0, 0.0]
        elsif category == 'POLITIQUE'
          label_values = [0.0, 1.0]
        end

        label = Java::OrgNd4jLinalgFactory::Nd4j.create(label_values.to_java(Java::float), [1,2].to_java(Java::int))

        datasets << DataSet.new(data, label)
      end

      break if datasets.size == 1000
    end

    puts "loaded #{num_row} samples from #{file}"

    datasets
  end
end
