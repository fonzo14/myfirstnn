java_import 'java.io.StringReader'
java_import 'edu.stanford.nlp.process.PTBTokenizer'

class Tokenizer
  PUNCTUATIONS = [':','?',',',';','.','...','``','%',"´",'/',"\\",'_','!',"`",'*','#','&','و','°']

  def initialize
    @factory = PTBTokenizer::PTBTokenizerFactory.newTokenizerFactory()
  end

  def tokenize(text)
    tokenizer = @factory.getTokenizer(StringReader.new(text))
    words     = tokenizer.tokenize
    # post process tokenizer
    words = words.flat_map { |w| w.word.split(/'/) }
      .map { |w| w.to_java(:string).toLowerCase }
      .reject { |w| w[0] == '-' && w[-1] == '-' }
      .reject { |w| PUNCTUATIONS.include?(w) }
      .reject { |w| w.size < 2 }
      .reject { |w| w.to_i > 0 }
    words
  end
end
