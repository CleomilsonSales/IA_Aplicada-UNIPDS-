import tf from '@tensorflow/tfjs-node';

async function trainModel(inputXs, outputYs) {
    const model = tf.sequential();

    /*
    primeiro camada de rede
    entrada de 7 posições (idade normalizada + 3 cores + 3 localizações)

    80 neuronios = esse valor porque tem pouca busca de treino
    quanto mais neuronio mais complexidade a rede aprende
    e consequentemente mais processamento


    a relu age como um filtro
    e como se ela deixasse apenas dados interessantes seguirem se chegou nesse neuronio é positivo
    se for zero ou negativo e descartado
    */

    model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu' }));

    /*
    saida: 3 neuronios
    um para cada categoria de pessoa (premium, medium, basic)

    softmax: transforma os valores de saida em probabilidades
     */
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));

    /*
    Adam: treinado pessoal moderno para redes reurais
    aprende com erro e acertos

    categoricalCrossentropy: compara o que o modelo acha (os scores de cada categoria)

    accuracy: mede a acuracia do modelo, ou seja, quantas vezes ele acertou a categoria correta

    */

    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    })

    // Treinamento do modelo
    /*
    verbose: desativa o log interno
    epochs: quantidade de vezes que vai rodar o dataset
    shuffle: embaralha os dados, para evitar viés
    */
    await model.fit(inputXs, outputYs, {
        verbose: 0,
        epochs: 100,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, log) => {
               //console.log(`Epoch ${epoch}: loss = ${log.loss}`);
            }
        }
    })

    return model
}

async function predict(model, pessoa){
    //transformando o array em tensor
    const tfInput = tf.tensor2d(pessoa)
    //fazer a predição (output sera um vator de 3 probabilidades)
    const pred = model.predict(tfInput)
    const predArray = await pred.array()    
    //console.log(`Predição: ${predArray}`)
    return predArray[0].map((prob, index) => ({prob, index}))
} 

//projeto precisou ser criado em wsl

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

//quantos mais dados melhor para o aprendizado
const model = await trainModel(inputXs, outputYs)

const pessoa = {
    nome: "Cleomilson",
    idade: 28,
    cor: "verde",
    localizacao: "Curitiba"
}

//normalizando a idade da nova pessoa pelo padrão do treino
//exemplo: idade_min = 25, idade_max = 40, então (28 - 25)/(40 - 25) = 0.2
const pessoaTensorNormalizado = [
    [
    0.2, //idade normalizada
    1, // azul
    0, // vermelho
    0, // verde
    0, // São Paulo
    1, // Rio
    0  // Curitiba
    ]
]

const predictions = await predict(model, pessoaTensorNormalizado) 
const results = predictions
    .sort((a, b) => b.prob - a.prob)
    .map(p => `${labelsNomes[p.index]} (${(p.prob * 100).toFixed(2)}%)`)
    .join('\n')

console.log(results)    