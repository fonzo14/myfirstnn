class SamplesLoader
  def initialize(vocabulary, tokenizer)
    @vocabulary = vocabulary
    @tokenizer  = tokenizer
  end

  def load(file)
    samples = []
    IO.foreach(File.join(NN_ROOT, 'data', file)) do |line|
      text, category = line.chomp.split("\t")
      data = SimpleMatrix.new(@vocabulary.size, 1)
      @tokenizer.tokenize(text).inject(Hash.new(0)) do |h,word|
        h[@vocabulary.to_id(word)] += 1
        h
      end.each do |word_id, word_count|
        data.set(word_id, word_count)
      end
      label = SimpleMatrix.new(1, 2)
      label.set(0, 0, category == 'SPORT' ? 1.0 : 0)
      label.set(0, 1, category == 'POLITIQUE' ? 1.0 : 0)
      samples << Sample.new(data, label)
    end
    puts "load #{samples.size} samples from #{file}"
    samples
  end
end
