let guessTextElement = undefined;
let strokeWeightSlider = undefined;
let selectedColor = 'black';
let session = undefined;

const champNames = ['Aatrox', 'Ahri', 'Akali', 'Akshan', 'Alistar', 'Ambessa', 'Amumu', 'Anivia', 'Annie', 'Aphelios', 'Ashe', 'AurelionSol', 'Aurora', 'Azir', 'Bard', 'Belveth', 'Blitzcrank', 'Brand', 'Braum', 'Briar', 'Caitlyn', 'Camille', 'Cassiopeia', 'Chogath', 'Corki', 'Darius', 'Diana', 'DrMundo', 'Draven', 'Ekko', 'Elise', 'Evelynn', 'Ezreal', 'Fiddlesticks', 'Fiora', 'Fizz', 'Galio', 'Gangplank', 'Garen', 'Gnar', 'Gragas', 'Graves', 'Gwen', 'Hecarim', 'Heimerdinger', 'Hwei', 'Illaoi', 'Irelia', 'Ivern', 'Janna', 'JarvanIV', 'Jax', 'Jayce', 'Jhin', 'Jinx', 'KSante', 'Kaisa', 'Kalista', 'Karma', 'Karthus', 'Kassadin', 'Katarina', 'Kayle', 'Kayn', 'Kennen', 'Khazix', 'Kindred', 'Kled', 'KogMaw', 'Leblanc', 'LeeSin', 'Leona', 'Lillia', 'Lissandra', 'Lucian', 'Lulu', 'Lux', 'Malphite', 'Malzahar', 'Maokai', 'MasterYi', 'Mel', 'Milio', 'MissFortune', 'MonkeyKing', 'Mordekaiser', 'Morgana', 'Naafiri', 'Nami', 'Nasus', 'Nautilus', 'Neeko', 'Nidalee', 'Nilah', 'Nocturne', 'Nunu', 'Olaf', 'Orianna', 'Ornn', 'Pantheon', 'Poppy', 'Pyke', 'Qiyana', 'Quinn', 'Rakan', 'Rammus', 'RekSai', 'Rell', 'Renata', 'Renekton', 'Rengar', 'Riven', 'Rumble', 'Ryze', 'Samira', 'Sejuani', 'Senna', 'Seraphine', 'Sett', 'Shaco', 'Shen', 'Shyvana', 'Singed', 'Sion', 'Sivir', 'Skarner', 'Smolder', 'Sona', 'Soraka', 'Swain', 'Sylas', 'Syndra', 'TahmKench', 'Taliyah', 'Talon', 'Taric', 'Teemo', 'Thresh', 'Tristana', 'Trundle', 'Tryndamere', 'TwistedFate', 'Twitch', 'Udyr', 'Urgot', 'Varus', 'Vayne', 'Veigar', 'Velkoz', 'Vex', 'Vi', 'Viego', 'Viktor', 'Vladimir', 'Volibear', 'Warwick', 'Xayah', 'Xerath', 'XinZhao', 'Yasuo', 'Yone', 'Yorick', 'Yuumi', 'Zac', 'Zed', 'Zeri', 'Ziggs', 'Zilean', 'Zoe', 'Zyra'];
const modelName = 'mobilenet';

const COLOURS = [
    'white',
    'lightGrey',
    'red',
    'orange',
    'rgb(245 245 0)',
    'lime',
    'cyan',
    'blue',
    'magenta',
    'pink',
    'peru',
    'black',
    'dimGrey',
    'darkRed',
    'chocolate',
    'gold',
    'green',
    'dodgerBlue',
    'midnightBlue',
    'purple',
    'paleVioletRed',
    'saddleBrown',
];


function setup() {
    createCanvas(512, 512);
    background(255);
    guessTextElement = createP('Guess: None');
    guessTextElement.style('font-size', '24px');
    for(let i = 0; i < COLOURS.length; i++) {
        const colorButton = createButton('');
        colorButton.size(50, 50);
        if ((i + 1) == parseInt(COLOURS.length / 2)) {
            createElement('br');
        }
        colorButton.style('background-color', COLOURS[i]);
        colorButton.mousePressed(() => {
            selectedColor = COLOURS[i];
        });
    }

    createElement('br');
    strokeWeightSlider = createSlider(5, 30, 10);
    strokeWeightSlider.size(300);

    ort.InferenceSession.create(modelName + '.onnx').then(s => {
        session = s;
        setInterval(() => {
            updateGuess();
        }, 1000);
    });
}

function softmax(arr) {
    const expArr = arr.map(x => Math.exp(x));
    const sumExp = expArr.reduce((sum, x) => sum + x, 0);
    const softmaxArr = expArr.map(x => x / sumExp);
    return softmaxArr;
}

function getMaxIndex(arr) {
    const max = Math.max(...arr);
    return arr.indexOf(max);
}

async function updateGuess() {
    const inputs = getInputData();
    const predictions = await runModel(inputs);
    const outputs = Array.from(predictions);
    const probabilities = softmax(outputs);
    const predictedIndex = getMaxIndex(probabilities);
    const predictedChampion = champNames[predictedIndex];
    const confidence = probabilities[predictedIndex];
    guessTextElement.html(`Guess: ${predictedChampion} Confidence: ${parseInt(confidence * 100)}%`)

}

function normalize(value, mean, std) {
    return ((value / 255) - mean) / std;
}

function getInputData() {
    let inputs = [];
    let img = get();
    img.resize(64, 64);
    img.loadPixels();

    const mean = [0.9568, 0.9540, 0.9552];
    const std = [0.1607, 0.1632, 0.1631];

    for(let i = 0; i < img.pixels.length; i+=4) {
        inputs.push(normalize(img.pixels[i], mean[0], std[0]));
    }
    for(let i = 1; i < img.pixels.length; i+=4) {
        inputs.push(normalize(img.pixels[i], mean[1], std[1]));
    }
    for(let i = 2; i < img.pixels.length; i+=4) {
        inputs.push(normalize(img.pixels[i], mean[2], std[2]));
    }

    return new Float32Array(inputs);
}

async function runModel(inputs) {
    const input = new ort.Tensor('float32', inputs, [1, 3, 64, 64]);
    const feeds = { 'input.1': input };
    const results = await session.run(feeds);
    if(modelName == 'mobilenet'){
        return results['414'].cpuData;
    } else {
        return results['495'].cpuData;
    }
}


function mouseDragged() {
    strokeWeight(strokeWeightSlider.value());
    if(selectedColor == 'rgb(245 245 0)') {
        stroke(245, 245, 0);
    } else {
        stroke(selectedColor);
    }
    line(pmouseX, pmouseY, mouseX, mouseY);
}
