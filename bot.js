/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
//const fetch = require('node-fetch');
const useLoader = require('@tensorflow-models/universal-sentence-encoder');
const tf = require('@tensorflow/tfjs-node');
global.fetch = require('node-fetch');

const d3 = require('d3');

tf.ENV.set('WEBGL_PACK', false);

const EMBEDDING_DIM = 512;

const DENSE_MODEL_URL = 'http://laka.ngrok.io/models/intent/model.json';
const METADATA_URL = 'http://laka.ngrok.io/models/intent/intent_metadata.json';

const modelUrls = {
  'bidirectional-lstm': 'http://laka.ngrok.io/models/bidirectional-tagger/model.json'
};

let use;
async function loadUSE() {
  if (use == null) {
    use = await useLoader.load();
  }
  return use;
}

let intent;
async function loadIntentClassifer(url) {
  if (intent == null) {
    intent = await tf.loadLayersModel(url);
  }
  return intent;
}

let intentMetadata;
async function loadIntentMetadata() {
  if (intentMetadata == null) {
    const resp = await fetch(METADATA_URL);
    intentMetadata = resp.json();
  }
  return intentMetadata;
}

exports.classify = async function (sentences) {
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

const THRESHOLD = 0.90;
exports.getClassificationMessage = async function (softmaxArr, inputText) {
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
        const model = "bidirectional-lstm";
        var location = await tagMessage(inputText, model);
        if (location.trim() != "") {
          const weatherMessage = await getWeather(location);;
          response = '⛅ ' + weatherMessage;
          //speak(weatherMessage);
        } else {
          response = '⛅';
        }
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

const taggers = {};
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

/**
 * Load metadata for a model.
 * @param {string} name Name of model. Should be a key in modelUrls
 */
async function loadMetadata(name) {
  const metadataUrl =
    modelUrls[name].replace('model.json', 'tagger_metadata.json');
  try {
    const resp = await fetch(metadataUrl);
    return resp.json();
  } catch (e) {
    // Could not load that model. This is not necessarily an error
    // as the user may not have trained all the available model types
    console.log(`Could not load "${name}" metadata`);
  }
}

/**
 * Load a number of models to allow the browser to cache them.
 */
async function loadTaggerModel() {
  const modelLoadPromises = Object.keys(modelUrls).map(loadTagger);
  return await Promise.all([loadUSE(), ...modelLoadPromises]);
}

/**
 * Split an input string into tokens, we use the same tokenization function
 * as we did during training.
 * @param {string} input
 *
 * @return {string[]}
 */
function tokenizeSentence(input) {
  return input.split(/\b/).map(t => t.trim()).filter(t => t.length !== 0);
}

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

    return location;
  }
}


async function getWeatherSearch(location) {
  const response = await fetch(`https://www.metaweather.com/api/location/search/?query=${location.trim()}`);

  const weatherSearch = response.json();


  return weatherSearch;
}

async function getWeather(location) {
  const weatherSearch = await getWeatherSearch(location);

  if (weatherSearch.length > 0) {
    const weatherResponse = await fetch(`https://www.metaweather.com/api/location/${weatherSearch[0].woeid}/`);
    const weather = await weatherResponse.json()

    return `The ${weather.location_type} of ${weather.title} is expecting ${weather.consolidated_weather[0].weather_state_name} today.`
  } else {
    return `I'm not smart enough to know weather data for ${location}`
  }
}

loadTaggerModel();
