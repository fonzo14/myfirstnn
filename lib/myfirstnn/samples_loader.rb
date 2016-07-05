class SamplesLoader
  def initialize(embeddings, tokenizer)
    @embeddings = embeddings
    @tokenizer  = tokenizer
  end

  def load(file)
    samples = []
    IO.foreach(File.join(NN_ROOT, 'data', file)) do |line|
      text, category = line.chomp.split("\t")
      data = SimpleMatrix.new(@embeddings.dimensions_count, 1)

      h = Hash.new(0)
      tokens = @tokenizer.tokenize(text).select { |word| @embeddings.include?(word) }
      tokens.each do |word|
        @embeddings.values(word).each_with_index do |v,i|
          h[i] += v
        end
      end

      h.each do |k,v|
        z = v / tokens.size
        data.set(k,z)
      end

      label = SimpleMatrix.new(2, 1)
      label.set(0, 0, category == 'SPORT' ? 1.0 : 0)
      label.set(1, 0, category == 'POLITIQUE' ? 1.0 : 0)
      samples << Sample.new([category, text].join(' -- '), data, label)
    end
    puts "loaded #{samples.size} samples from #{file}"
    samples
  end
end
