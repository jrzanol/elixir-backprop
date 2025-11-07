defmodule OneHotEncoder do
  @moduledoc """
  Implementa a codificação One-Hot para variáveis categóricas.
  """

  defstruct categorical_indices: [], categories_map: %{}

  def new(categorical_indices) do
    %OneHotEncoder{categorical_indices: categorical_indices}
  end

  def fit(%OneHotEncoder{categorical_indices: indices} = encoder, data) do
    categories_map = build_categories_map(data, indices)
    %{encoder | categories_map: categories_map}
  end

  def transform(%OneHotEncoder{categorical_indices: indices, categories_map: categories_map}, data) do
    Enum.map(data, fn row ->
      encode_row(row, indices, categories_map)
    end)
  end

  defp build_categories_map(data, categorical_indices) do
    categorical_indices
    |> Enum.map(fn idx ->
      unique_values =
        data
        |> Enum.map(&Enum.at(&1, idx))
        |> Enum.uniq()
        |> Enum.sort()
      {idx, unique_values}
    end)
    |> Map.new()
  end

  defp encode_row(row, categorical_indices, categories_map) do
    row
    |> Enum.with_index()
    |> Enum.flat_map(fn {value, idx} ->
      if idx in categorical_indices do
        categories = Map.get(categories_map, idx)
        encode_value(value, categories)
      else
        [value]
      end
    end)
  end

  defp encode_value(value, categories) do
    Enum.map(categories, fn category ->
      if value == category, do: 1.0, else: 0.0
    end)
  end
end
