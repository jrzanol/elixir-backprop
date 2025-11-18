import Bitwise

defmodule XorShift64Star do
  @mask 0xFFFFFFFFFFFFFFFF
  @mul  2685821657736338717

  def next(state) when state != 0 do
    x = state
    x = bxor(x, x >>> 12)
    x = bxor(x, (x <<< 25) &&& @mask)
    x = bxor(x, x >>> 27)
    x = x &&& @mask
    rem(x * @mul, @mask + 1)
  end

  # Retorna float ∈ [0,1)
  def uniform_float(state) do
    v = next(state)
    # Usa os 53 bits mais significativos para gerar float
    vfloat = (v >>> 11) / :math.pow(2, 53)
    {v, vfloat}
  end
end
