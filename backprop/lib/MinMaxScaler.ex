defmodule MinMaxScaler do
  @moduledoc """
  Implementa a normalização Min-Max para features numéricas.
  """

  defstruct mins: [], maxs: []

  def new do
    %MinMaxScaler{}
  end

  def fit(%MinMaxScaler{} = scaler, data) do
    n_features = length(hd(data))

    {mins, maxs} =
      Enum.reduce(0..(n_features - 1), {[], []}, fn idx, {mins_acc, maxs_acc} ->
        column = Enum.map(data, &Enum.at(&1, idx))
        min_val = Enum.min(column)
        max_val = Enum.max(column)
        {mins_acc ++ [min_val], maxs_acc ++ [max_val]}
      end)

    %{scaler | mins: mins, maxs: maxs}
  end

  def transform(%MinMaxScaler{mins: mins, maxs: maxs}, data) do
    Enum.map(data, fn row ->
      row
      |> Enum.with_index()
      |> Enum.map(fn {value, idx} ->
        min_val = Enum.at(mins, idx)
        max_val = Enum.at(maxs, idx)
        scale_value(value, min_val, max_val)
      end)
    end)
  end

  defp scale_value(_value, min_val, max_val) when max_val == min_val, do: 0.0
  defp scale_value(value, min_val, max_val), do: (value - min_val) / (max_val - min_val)
end
