defmodule ReadCsv do
  @moduledoc """
  Implementa a leitura e o parsing de arquivos CSV.
  """
  def read(filename) do
    case File.read(filename) do
      {:ok, content} ->
        content
        |> String.trim()
        |> String.split("\n")
        |> Enum.drop(1) # Skip header
        |> Enum.map(&parse_csv_line/1)
        |> Enum.filter(fn x -> x != nil end)
      {:error, reason} ->
        IO.puts("Error reading file #{filename}: #{reason}")
        []
    end
  end

  defp parse_csv_line(line) do
    try do
      values = String.split(line, ",")
              |> Enum.map(&String.trim/1)
              |> Enum.map(&parse_float/1)

      case values do
        [] -> nil
        [_] -> nil
        _ ->
          inputs = Enum.drop(values, -1)
          target = [List.last(values)]
          {inputs, target}
      end
    rescue
      _ -> nil
    end
  end

  defp parse_float(str) do
    case Float.parse(str) do
      {float, _} -> float
      :error ->
        case Integer.parse(str) do
          {int, _} -> int * 1.0
          :error -> 0.0
        end
    end
  end
end
