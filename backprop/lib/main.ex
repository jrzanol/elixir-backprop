alias ReadCsv
alias MinMaxScale
alias OneHotEncoder
alias MLPClassifier
alias ModelSelection

# Leitura do CSV contendo o dataset:
dataset = ReadCsv.read("heart.csv")
{x, y_raw} = Enum.unzip(dataset)

y = Enum.map(y_raw, fn yval ->
  val = if is_list(yval), do: hd(yval), else: yval
  trunc(val)
end)

# Separação do dataset de treino e teste:
{x_train, x_test, y_train, y_test} = ModelSelection.train_test_split(x, y, 0.33, true)

# One-Hot Encoding para os dados categóricos:
encoder = OneHotEncoder.new([1, 2, 6, 8, 10])
encoder = OneHotEncoder.fit(encoder, x_train)
x_train_encoded = OneHotEncoder.transform(encoder, x_train)
x_test_encoded = OneHotEncoder.transform(encoder, x_test)

# Normaliza as features:
scaler = MinMaxScaler.new()
scaler = MinMaxScaler.fit(scaler, x_train_encoded)
x_train_norm = MinMaxScaler.transform(scaler, x_train_encoded)
x_test_norm = MinMaxScaler.transform(scaler, x_test_encoded)

# Cria e Treina o modelo MLP:
mlp = MLPClassifier.new([length(hd(x_train_norm)), 8, 1])
mlp_trained = MLPClassifier.fit(mlp, x_train_norm, y_train, 1000, 0.001)

# Realiza predições no conjunto de teste:
preds = Enum.map(x_test_norm, &MLPClassifier.predict(mlp_trained, &1))

# Calcula a taxa de acerto
correct = Enum.zip(preds, y_test) |> Enum.count(fn {pred, actual} -> pred == actual end)
accuracy = correct / length(y_test) * 100

IO.inspect(y_test, label: "Valor Corretos:")
IO.inspect(preds, label: "Predições:")

IO.puts("Taxa de acerto: #{Float.round(accuracy, 2)}%")
