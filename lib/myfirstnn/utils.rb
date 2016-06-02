module Utils
  class << self
    def tanh(matrix)
      new_matrix(matrix) { |value| java.lang.Math.tanh(value) }
    end

    def sigmoid(matrix)
      new_matrix(matrix) { |value| (1 / (1 + java.lang.Math.exp(-value))) }
    end

    def softmax(matrix)
      nm  = new_matrix(matrix) { |value| java.lang.Math.exp(value) }
      sum = nm.elementSum()
      new_matrix(nm) { |value| value / sum }
    end

    def new_matrix(matrix)
      new_matrix = SimpleMatrix.new(matrix.numRows, matrix.numCols)
      matrix.numRows.times do |i|
        matrix.numCols.times do |j|
          new_matrix.set(i, j, yield(matrix.get(i,j)))
        end
      end
      new_matrix
    end
  end
end
