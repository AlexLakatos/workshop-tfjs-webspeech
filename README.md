# jsconf.asia workshop - How to Build Talking Websites with the Web Speech API and TensorFlow.js

The Tensorflow.js example uses the universal sentence encoder to train two text classification models.

1. An 'intent' classifier that classifies sentences into categories representing
user intent for a query.
2. A token tagger, that classifies tokens within a weather releated query to
identify location related tokens.

## Setup and Installation

Note: These instructions use `yarn`, but you can use `npm run` instead if you
do not have `yarn` installed.

Install dependencies

```
yarn
```

## Preparing training data

There are four npm/yarn scripts listed in package.json for preparing the training data. Each writes out one of more new files.

The two scripts needed to train the intent classifier are:

1. `yarn raw-to-csv`: Converts the raw data into a csv format
2. `yarn csv-to-tensors`: Converts the strings in the CSV created in step 1 into tensors.

The two scripts needed to train the token tagger are:

1. `yarn raw-to-tagged-tokens`: Extracts tokens from sentences in the original data and tags each token with a category
2. `yarn tokens-to-embeddings`: embeds the tokens from the queries using the universal sentence encoder and writes out a look-up-table.

You can run all four of these commands with

```
yarn prep-data
```

You only need to do this once. This process can take 2-5 minutes on the smaller data sets and up to an hour on the full data set. The output of these scripts will be written to the `training/data` folder.

## Train the models

To train the intent classifier model run:

```
yarn train-intent
```

To train the token tagging model run:

```
yarn train-tagger
```

Each of these scripts take multiple options, look at `training/train-intent.js` and `training/train-tagger.js` for details.

These scripts will output model artifacts in the `training/models` folder.

You can run all two of these commands with

```
yarn train
```

## Run the app

Once the models are trained you can use the following command to run the demo app

```
yarn workshop-app
```


## Build the app

### Load the models

```
git checkout -b load-tensor
```

```javascript
async function loadIntentClassifer(url) {
  if (intent == null) {
    intent = await tf.loadLayersModel(url);
  }
  return intent;
}
```

```javascript
/**
 * Load a custom trained token tagger model.
 * @param {string} name Type of model to load. Should be a key in modelUrls
 */
async function loadTagger(name) {
  if (taggers[name] == null) {
    const url = modelUrls[name];
    try {
      taggers[name] = await tf.loadLayersModel(url);
    } catch (e) {
      // Could not load that model. This is not necessarily an error
      // as the user may not have trained all the available model types
      console.log(`Could not load "${name}" model`);
    }
  }
  return taggers[name];
}
```

```javascript
/**
 * Load a number of models to allow the browser to cache them.
 */
async function loadTaggerModel() {
  const modelLoadPromises = Object.keys(modelUrls).map(loadTagger);
  return await Promise.all([loadUSE(), ...modelLoadPromises]);
}
```


```javascript
window.addEventListener('load', function() {
  loadTaggerModel();
  setupListeners();
});
```

### Add intent classification

```
git checkout -b add-intents
```

```javascript
async function classify(sentences) {
  const [use, intent, metadata] = await Promise.all(
    [loadUSE(), loadIntentClassifer(DENSE_MODEL_URL), loadIntentMetadata()]);

  const {
    labels
  } = metadata;
  const activations = await use.embed(sentences);

  const prediction = intent.predict(activations);

  const predsArr = await prediction.array();
  const preview = [predsArr[0].slice()];
  preview.unshift(labels);
  console.table(preview);

  tf.dispose([activations, prediction]);

  return predsArr[0];
}
```

```javascript
async function getClassificationMessage(softmaxArr, inputText) {
  const {
    labels
  } = await loadIntentMetadata();
  const max = Math.max(...softmaxArr);
  const maxIndex = softmaxArr.indexOf(max);
  const intentLabel = labels[maxIndex];

  if (max < THRESHOLD) {
    return '¯\\_(ツ)_/¯';
  } else {
    let response;
    switch (intentLabel) {
      case 'GetWeather':
        response = '⛅'
        break;

      case 'PlayMusic':
        response = '🎵🎺🎵';
        break;

      default:
        response = '?';
        break;
    }
    return response;
  }
}
```

```javascript
async function sendMessage(inputText) {
  ...

  // Classify the text
  const classification = await classify([inputText]);


  // Add the response to the chat window
  const response = await getClassificationMessage(classification, inputText);
  appendMessage(response, 'bot', msgId);

  ...
}
```


### Add message tagging

```
git checkout -b add-tagging
```

```javascript
case 'GetWeather':
  const model = "bidirectional-lstm";
  var location = await tagMessage(inputText, model);
  response = '⛅ ' + location.trim();
  break;
```

```javascript
/**
 * Tokenize a sentence and tag the tokens.
 *
 * @param {string} sentence sentence to tag
 * @param {string} model name of model to use
 *
 * @return {Object} dictionary of tokens, model outputs and embeddings
 */
async function tagTokens(sentence, model = 'bidirectional-lstm') {
  const [use, tagger, metadata] =
  await Promise.all([loadUSE(), loadTagger(model), loadMetadata(model)]);
  const {
    labels,
    sequenceLength
  } = metadata;


  let tokenized = tokenizeSentence(sentence);
  if (tokenized.length > sequenceLength) {
    console.warn(
      `Input sentence has more tokens than max allowed tokens ` +
      `(${sequenceLength}). Extra tokens will be dropped.`);
  }
  tokenized = tokenized.slice(0, sequenceLength);
  const activations = await use.embed(tokenized);

  // get prediction
  const prediction = tf.tidy(() => {
    // Make an input tensor of [1, sequence_len, embedding_size];
    const toPad = sequenceLength - tokenized.length;

    const padTensors = tf.ones([toPad, EMBEDDING_DIM]);
    const padded = activations.concat(padTensors);

    const batched = padded.expandDims();
    return tagger.predict(batched);
  });


  // Prediction data
  let predsArr = (await prediction.array())[0];

  // Add padding 'tokens' to the end of the values that will be displayed
  // in the UI. These are there for illustration.
  if (tokenized.length < sequenceLength) {
    tokenized.push(labels[2]);
    predsArr = predsArr.slice(0, tokenized.length);
  }

  // Add an extra activation to illustrate the padding inputs in the UI.
  // This is added for illustration.
  const displayActivations =
    tf.tidy(() => activations.concat(tf.ones([1, EMBEDDING_DIM])));
  const displayActicationsArr = await displayActivations.array();

  tf.dispose([activations, prediction, displayActivations]);

  return {
    tokenized: tokenized,
    tokenScores: predsArr,
    tokenEmbeddings: displayActicationsArr,
  };
}
```

```javascript
async function tagMessage(inputText, model) {
  if (inputText != null && inputText.length > 0) {
    const result = await tagTokens(inputText, model);
    const {
      tokenized,
      tokenScores,
      tokenEmbeddings
    } = result;
    const metadata = await loadMetadata(model);
    const {
      labels
    } = metadata;
    const location = tokenScores
      .map((scores, index) => {
        const maxIndex = scores.indexOf(Math.max(...scores));
        if (maxIndex === 1) {
          return tokenized[index]
        }
      })
      .join('\ ')

    console.log(location);

    return location;
  }
}
```

### Add the weather api for tagged location

```
git checkout -b add-weather-api
```

```javascript
case 'GetWeather':
        const model = "bidirectional-lstm";
        var location = await tagMessage(inputText, model);
        if (location.trim() != "") {
          const weatherMessage = await getWeather(location);;
          response = '⛅ ' + weatherMessage;
        } else {
          response = '⛅';
        }
        break;
```

```javascript
async function getWeatherSearch(location) {
  const response = await fetch(`https://cors-anywhere.herokuapp.com/https://www.metaweather.com/api/location/search/?query=${location.trim()}`);

  const weatherSearch = response.json();


  return weatherSearch;
}
```

```javascript
async function getWeather(location) {
  const weatherSearch = await getWeatherSearch(location);

  if (weatherSearch.length > 0) {
    const weatherResponse = await fetch(`https://cors-anywhere.herokuapp.com/https://www.metaweather.com/api/location/${weatherSearch[0].woeid}/`);
    const weather = await weatherResponse.json()

    return `The ${weather.location_type} of ${weather.title} is expecting ${weather.consolidated_weather[0].weather_state_name} today.`
  } else {
    return `I'm not smart enough to know weather data for ${location}`
  }
}
```

### Visualize tokenization

```
git checkout -b display-tokenization
```

```javascript
/**
 * Render the tokens
 *
 * @param {string[]} tokens the tokens
 * @param {Array.number[]} tokenScores model scores for each token
 * @param {Array.number[]} tokenEmbeddings token embeddings
 * @param {string} model name of model
 */
async function displayTokenization(tokens, tokenScores, tokenEmbeddings, model) {
  const resultsDiv = document.createElement('div');
  resultsDiv.classList = `tagging`;
  resultsDiv.innerHTML = `<p class="model-type ${model}">${model}</p>`;

  displayTokens(tokens, resultsDiv);
  displayEmbeddingsPlot(tokenEmbeddings, resultsDiv);
  displayTags(tokenScores, resultsDiv, model);

  document.getElementById('taggings').prepend(resultsDiv);
}

/**
 * Render the tokens.
 *
 * @param {string[]} tokens tokens to display
 * @param {HTMLElement} parentEl parent element
 */
function displayTokens(tokens, parentEl) {
  const tokensDiv = document.createElement('div');
  tokensDiv.classList = `tokens`;
  tokensDiv.innerHTML =
    tokens.map(token => `<div class="token">${token}</div>`).join('\n');
  parentEl.appendChild(tokensDiv);
}

const embeddingCol =
  d3.scaleSequential(d3.interpolateSpectral).domain([-0.075, 0.075]);
embeddingCol.clamp(true);

/**
 * Display an illustrative representation of the embeddings values
 * @param {*} embeddings
 * @param {*} parentEl
 */
function displayEmbeddingsPlot(embeddings, parentEl) {
  const embeddingDiv = document.createElement('div');
  embeddingDiv.classList = `embeddings`;

  embeddingDiv.innerHTML =
    embeddings
    .map(embedding => {
      // Note that this slice is arbitraty as the plot is only meant to
      // be illustrative.
      const embeddingValDivs = embedding.slice(0, 340).map(val => {
        return `<div class="embVal" ` +
          `style="background-color:${embeddingCol(val)} "` +
          `title="${val}"` +
          `></div>`;
      });

      return `<div class="embedding">` +
        `${embeddingValDivs.join('\n')}</div>`;
    })
    .join('\n');

  parentEl.appendChild(embeddingDiv);
}

/**
 *
 * @param {*} tokenScores
 * @param {*} parentEl
 * @param {*} modelName
 */
async function displayTags(tokenScores, parentEl, modelName) {
  const metadata = await loadMetadata(modelName);
  const {
    labels
  } = metadata;
  let location = "";

  const tagsDiv = document.createElement('div');
  tagsDiv.classList = `tags`;
  tagsDiv.innerHTML =
    tokenScores
    .map((scores, index) => {
      const maxIndex = scores.indexOf(Math.max(...scores));
      const token = labels[maxIndex];
      const tokenScore = (scores[maxIndex] * 100).toPrecision(3);
      return `<div class="tag ${token}">` +
        `&nbsp;&nbsp;${token.replace(/__/g, '')}<sup>` +
        `${tokenScore}%</sup></div>`;
    })
    .join('\n');

  if (location != "") {
    appendMessage(location, 'bot', messageId - 1);
  }
  parentEl.appendChild(tagsDiv);
}
```

```javascript
async function tagMessage(inputText, model) {
  ...
    displayTokenization(tokenized, tokenScores, tokenEmbeddings, model);
  ...

    return location;
  }
```

### Add Speech Synthesis

```
git checkout -b add-speech-synthesis
```

```javascript
function speak(message) {
  let utterance = new SpeechSynthesisUtterance();

  utterance.text = message;
  utterance.rate = 5;
  utterance.pitch = 2;
  utterance.lang = "en-GB";

  speechSynthesis.speak(utterance);
}
```


```javascript
case 'GetWeather':
        const model = "bidirectional-lstm";
        var location = await tagMessage(inputText, model);
        if (location.trim() != "") {
          const weatherMessage = await getWeather(location);;
          response = '⛅ ' + weatherMessage;
          speak(weatherMessage);
        } else {
          response = '⛅';
        }
```

### Add Speech Recognition

```
git checkout -b add-speech-recognition
```

```javascript
function ask() {
  var SpeechRecognition = SpeechRecognition || webkitSpeechRecognition;
  var recognition = new SpeechRecognition();

  recognition.lang = "en-US";
  recognition.interimResults = false;
  recognition.start();

  recognition.addEventListener('result', (e) => {
    let last = e.results.length - 1;
    let text = e.results[last][0].transcript;
    console.log(text);

    sendMessage(text);

    recognition.stop();
  });
}
```

```javascript
function setupListeners() {
  ...
  speech.addEventListener('click', ask, false);
}
```
