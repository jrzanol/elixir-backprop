defmodule ModelSelection do
  @moduledoc """
  Implementa as funções para a separação do dataset em treino e teste.
  """

  def train_test_split(x, y, test_size \\ 0.25, shuffle \\ true) do
    n = length(x)

    indices =
      if shuffle do
        Enum.shuffle(0..(n - 1))
      else
        Enum.to_list(0..(n - 1))
      end

    test_count =
      cond do
        is_float(test_size) -> round(n * test_size)
        is_integer(test_size) -> test_size
        true -> round(n * 0.25)
      end

    {test_idx, train_idx} = Enum.split(indices, test_count)

    x_test = Enum.map(test_idx, &Enum.at(x, &1))
    y_test = Enum.map(test_idx, &Enum.at(y, &1))
    x_train = Enum.map(train_idx, &Enum.at(x, &1))
    y_train = Enum.map(train_idx, &Enum.at(y, &1))
    {x_train, x_test, y_train, y_test}
  end
end
