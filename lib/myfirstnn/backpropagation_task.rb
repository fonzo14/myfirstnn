class BackpropagationTask
  include Java::JavaUtilConcurrent::Callable

  def initialize(w_hidden, b_hidden, w_output, b_output, sample)
    @w_hidden = w_hidden
    @b_hidden = b_hidden
    @w_output = w_output
    @b_output = b_output

    @sample = sample
  end

  def call
    backpropagation
  end

  private
  def backpropagation
    # backpropagation implementation

    # feedforwarding sample
    # keep the intermediate value's computation :
    # z = weighted input layer l
    # a = activation layer l
    # so that we can compute the error/gradients
    hidden_z = @w_hidden.mult(@sample.data).plus(@b_hidden)
    hidden_a = Utils::tanh(hidden_z)

    output_z = @w_output.mult(hidden_a).plus(@b_output)
    output_a = Utils::softmax(output_z)

    # compute the output error / delta
    # delta = loss_derivative(output_a, sample.label).elementMult(softmax_derivative(output_z))
    # looks like derivative of the cross-entropy cost function for the softmax function is just :
    delta_output = output_a.minus(@sample.label)

    # output biais delta = delta
    delta_b_output = delta_output

    # output weight delta
    delta_w_output = delta_output.mult(hidden_a.transpose)

    # hidden layer
    delta_hidden   = @w_output.transpose.mult(delta_output).elementMult(Utils::tanh_derivative(hidden_z))

    delta_b_hidden = delta_hidden
    delta_w_hidden = delta_hidden.mult(@sample.data.transpose)

    [delta_w_output, delta_b_output, delta_w_hidden, delta_b_hidden]
  end
end
