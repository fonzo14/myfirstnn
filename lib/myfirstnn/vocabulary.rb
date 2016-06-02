class Vocabulary
  UNK = 'unk'

  attr_reader :size

  def initialize(size, min_count, file)
    @text2id = { UNK => 0 }
    @id2text = { 0 => UNK }

    IO.foreach(file) do |line|
      word_id, word_text, word_count = line.chomp.split("\t")
      if (word_count.to_i >= min_count) && @text2id.size < size
        @text2id[word_text]    = word_id.to_i
        @id2text[word_id.to_i] = word_text
      end
    end

    puts "vocabulary loaded #{@text2id.size} words"

    @size = @text2id.size
  end

  def to_text(word_id)
    @id2text[word_id] || UNK
  end

  def to_id(word_text)
    @text2id[word_text] || 0
  end
end
