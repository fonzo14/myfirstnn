java_import 'java.io.StringReader'
java_import 'edu.stanford.nlp.ling.DocumentReader'
java_import 'edu.stanford.nlp.international.french.process.FrenchTokenizer'
java_import 'edu.stanford.nlp.process.PTBTokenizer'

class Tokenizer
  def initialize
    @factory = PTBTokenizer::PTBTokenizerFactory.newTokenizerFactory()
  end

  def tokenize(text)
    tokenizer = @factory.getTokenizer(StringReader.new(text))
    words     = tokenizer.tokenize
    # post process tokenizer
    words     = words.flat_map { |w| w.word.split(/'|-/) }.map(&:downcase)
    words
  end
end
