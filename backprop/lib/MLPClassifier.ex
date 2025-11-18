alias XorShift64Star

defmodule MLPClassifier do
  @moduledoc """
  Implementa o Modelo de Classificação usando Perceptron Multicamadas (MLP).
  """

  defstruct weights: [], biases: [], layers: []

  def new(layers, seed) do
    rng0 = seed

    # Gera os pesos iniciais aleatoriamente.
    {weights, _} =
      Enum.chunk_every(layers, 2, 1, :discard)
      |> Enum.map_reduce(rng0, fn [a, b], rng ->
        limit = :math.sqrt(2.0 / a)

        Enum.map_reduce(1..b, rng, fn _, rng2 ->
          Enum.map_reduce(1..a, rng2, fn _, rng3 ->
            {rng4, u} = XorShift64Star.uniform_float(rng3)

            # random in [-1, 1]
            val = (u * 2 - 1) * limit

            {val, rng4}
          end)
        end)
        |> (fn {row_values, new_rng} -> {row_values, new_rng} end).()
      end)

    biases = Enum.map(tl(layers), fn b -> for _ <- 1..b, do: 0.0 end)

    %MLPClassifier{
      layers: layers,
      weights: weights,
      biases: biases
    }
  end

  def predict(model, x) do
    {activations, _} = forward(model, x)
    List.last(activations) |> Enum.map(&round/1) |> hd()
  end
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

  def fit(model, _x_train, _y_train, epochs, _lr) when epochs == 0, do: model
  def fit(model, x_train, y_train, epochs, lr) do
    updated =
      Enum.zip(x_train, y_train)
      |> Enum.reduce(model, fn {x, y}, m ->
        {activations, zs} = forward(m, x)
        {nabla_w, nabla_b} = backprop(m, activations, zs, y)
        update_params(m, nabla_w, nabla_b, lr)
      end)

    if rem(epochs, 100) == 0 do
      loss = calculate_loss(updated, x_train, y_train)
      IO.puts("Época #{1000 - epochs + 1}, Loss: #{Float.round(loss, 4)}")
    end

    fit(updated, x_train, y_train, epochs - 1, lr)
  end
  defp backprop(model, activations, zs, y_true) do
    y_list = if is_list(y_true), do: y_true, else: [y_true]
    y_pred = List.last(activations)
    last_z = List.last(zs)

    delta = Enum.zip([y_pred, y_list, last_z])
            |> Enum.map(fn {a, y, z} -> (a - y) * sigmoid_prime(z) end)

    nabla_b = [delta]
    nabla_w = [Enum.map(delta, fn d ->
      Enum.map(Enum.at(activations, -2), fn a -> d * a end)
    end)]

    propagate_error(model.weights, activations, zs, delta, nabla_w, nabla_b)
  end
  defp propagate_error(weights, activations, zs, delta, nabla_w, nabla_b) do
    num_layers = length(weights)

    Enum.reduce((num_layers - 1)..1//-1, {delta, nabla_w, nabla_b}, fn l, {curr_delta, nw, nb} ->
      z = Enum.at(zs, l - 1)
      sp = Enum.map(z, &sigmoid_prime/1)
      w_next = Enum.at(weights, l)

      new_delta = Enum.with_index(sp)
                  |> Enum.map(fn {sp_val, i} ->
                    error = Enum.zip(w_next, curr_delta)
                            |> Enum.map(fn {neuron_w, d} -> Enum.at(neuron_w, i) * d end)
                            |> Enum.sum()
                    sp_val * error
                  end)

      new_nabla_w = Enum.map(new_delta, fn d ->
        Enum.map(Enum.at(activations, l - 1), fn a -> d * a end)
      end)

      {new_delta, [new_nabla_w | nw], [new_delta | nb]}
    end)
    |> then(fn {_, nw, nb} -> {nw, nb} end)
  end
  defp update_params(model, nabla_w, nabla_b, lr) do
    new_weights = Enum.zip(model.weights, nabla_w)
                  |> Enum.map(fn {layer_w, layer_nw} ->
                    Enum.zip(layer_w, layer_nw)
                    |> Enum.map(fn {neuron_w, neuron_nw} ->
                      Enum.zip(neuron_w, neuron_nw)
                      |> Enum.map(fn {w, nw} -> w - lr * nw end)
                    end)
                  end)

    new_biases = Enum.zip(model.biases, nabla_b)
                 |> Enum.map(fn {layer_b, layer_nb} ->
                   Enum.zip(layer_b, layer_nb)
                   |> Enum.map(fn {b, nb} -> b - lr * nb end)
                 end)

    %{model | weights: new_weights, biases: new_biases}
  end
  defp calculate_loss(model, x_train, y_train) do
    errors = Enum.zip(x_train, y_train)
             |> Enum.map(fn {x, y} ->
               {activations, _} = forward(model, x)
               output = List.last(activations) |> hd()
               y_val = if is_list(y), do: hd(y), else: y
               (output - y_val) * (output - y_val)
             end)

    Enum.sum(errors) / length(errors)
  end

  defp sigmoid(x), do: 1.0 / (1.0 + :math.exp(-x))
  defp sigmoid_prime(z), do: sigmoid(z) * (1.0 - sigmoid(z))
  defp dot_product(a, b), do: Enum.zip(a, b) |> Enum.map(fn {x, y} -> x * y end) |> Enum.sum()
end
