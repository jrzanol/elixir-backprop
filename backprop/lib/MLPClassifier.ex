defmodule MLPClassifier do
  defstruct weights: [], biases: [], layers: []

  def new(layers), do: %__MODULE__{layers: layers}

  def fit(model, x_train, y_train, epochs \\ 1000, lr \\ 0.1) do
    {weights, biases} = initialize_params(model.layers)
    train_loop(%{model | weights: weights, biases: biases}, x_train, y_train, epochs, lr)
  end

  defp initialize_params(layers) do
    weights =
      Enum.chunk_every(layers, 2, 1, :discard)
      |> Enum.map(fn [a, b] ->
        for _ <- 1..b, do: (for _ <- 1..a, do: :rand.uniform() * 0.1)
      end)

    biases =
      Enum.map(tl(layers), fn b ->
        for _ <- 1..b, do: 0.0
      end)

    {weights, biases}
  end

  defp train_loop(model, _, _, 0, _), do: model
  defp train_loop(model, x_train, y_train, epochs, lr) do
    updated =
      Enum.zip(x_train, y_train)
      |> Enum.reduce(model, fn {x, y}, m ->
        {activations, zs} = forward(m, x)
        nabla = backprop(m, activations, zs, y)
        update_params(m, nabla, lr)
      end)

    train_loop(updated, x_train, y_train, epochs - 1, lr)
  end

  def predict(model, x) do
    {activations, _} = forward(model, x)
    List.last(activations) |> Enum.map(&round/1)
  end

  # Forward pass
  defp forward(model, input) do
    Enum.zip(model.weights, model.biases)
    |> Enum.reduce({[input], []}, fn {w, b}, {acts, zs} ->
      z =
        Enum.map(Enum.zip(w, b), fn {weights_row, bias} ->
          bias + dot_product(weights_row, List.last(acts))
        end)

      a = Enum.map(z, &sigmoid/1)
      {acts ++ [a], zs ++ [z]}
    end)
  end

  # Backpropagation
  defp backprop(_model, activations, zs, y_true) do
    y_pred = List.last(activations)
    delta = Enum.zip(y_pred, y_true) |> Enum.map(fn {a, y} -> (a - y) * sigmoid_prime(a) end)
    {delta, activations, zs}
  end

  # Atualização simplificada (para exemplo didático)
  defp update_params(model, {_, _, _}, _lr), do: model

  defp sigmoid(x), do: 1.0 / (1.0 + :math.exp(-x))
  defp sigmoid_prime(x), do: x * (1.0 - x)
  defp dot_product(a, b), do: Enum.zip(a, b) |> Enum.map(fn {x, y} -> x * y end) |> Enum.sum()
end
