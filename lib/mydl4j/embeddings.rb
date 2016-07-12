class Embeddings
  UNK = 'unk'

  attr_reader :dimensions_count

  def initialize(file)
    @embeddings = {}

    IO.foreach(file).with_index do |line,i|
      if i < 1002
        #puts i if (i % 5000).zero?

        if i.zero?
          @size, @dimensions_count = line.chomp.split("\s").map(&:to_i)

          puts "Embeddings loading #{@size} words / #{@dimensions_count} dimensions"
        else
          token, *values = line.chomp.split("\s")
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
