class Embeddings
  UNK = 'unk'

  attr_reader :dimensions_count

  def initialize(file)
    @text2id    = { UNK => 0 }
    @id2text    = { 0 => UNK }
    @embeddings = {}

    IO.foreach(file).with_index do |line,i|
      if i < 1000000002
        #puts i if (i % 5000).zero?

        if i.zero?
          @size, @dimensions_count = line.chomp.split("\s").map(&:to_i)

          puts "Embeddings loading #{@size} words / #{@dimensions_count} dimensions"
        else
          token, *values = line.chomp.split("\s")
          @text2id[token] = i
          @id2text[i] = token
          @embeddings[token] = values.map(&:to_f)
        end
      end
    end

    puts "embeddings loaded"
  end

  def include?(token)
    @embeddings.key?(token)
  end

  def values(token)
    @embeddings[token]
  end
end
